#!/usr/bin/env python3
"""Inference engine for the Mason transformer.

Handles token-by-token generation with temperature, top-k, top-p sampling,
and tool-call interception.

Usage:
    python generate.py --prompt "Estimate 500 LF of 8-inch sewer"
    python generate.py --interactive
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent

# Explicit imports to avoid name collisions with the estimator's model.py
def _import_local(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cfg_mod = _import_local("transformer_config", SCRIPT_DIR / "config.py")
_tok_mod = _import_local("transformer_tokenizer", SCRIPT_DIR / "tokenizer.py")
_mdl_mod = _import_local("transformer_model", SCRIPT_DIR / "model.py")

Config = _cfg_mod.Config
BPETokenizer = _tok_mod.BPETokenizer
MasonTransformer = _mdl_mod.MasonTransformer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MasonEngine:
    """Inference engine: generates text, intercepts tool calls, executes tools."""

    def __init__(self, model_path: str = None, device=None):
        self.device = device or get_device()
        self.cfg = Config()

        # Load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(str(SCRIPT_DIR / self.cfg.tokenizer_path))
        self.cfg.vocab_size = self.tokenizer.vocab_size

        # Load model
        if model_path is None:
            model_path = str(SCRIPT_DIR / self.cfg.best_model_path)

        if Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

            # Rebuild Config from saved checkpoint values so architecture always matches
            if "cfg" in ckpt:
                saved = ckpt["cfg"]
                for key, val in saved.items():
                    if hasattr(self.cfg, key):
                        try:
                            setattr(self.cfg, key, val)
                        except Exception:
                            pass

            self.model = MasonTransformer(self.cfg).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", "?")
            print(f"Loaded model from {model_path} (step {step}, n_embd={self.cfg.n_embd}, vocab={self.cfg.vocab_size})")
        else:
            self.model = MasonTransformer(self.cfg).to(self.device)
            print(f"WARNING: No model found at {model_path}, using random weights")
        self.model.eval()

        # Tool-call special token IDs
        self.tool_call_id = self.tokenizer.special_tokens.get("<|tool_call|>")
        self.tool_result_id = self.tokenizer.special_tokens.get("<|tool_result|>")
        self.end_id = self.tokenizer.end_id

    # ---- tool execution ----------------------------------------------------

    def execute_tool(self, tool_text: str) -> str:
        """Parse and execute a tool call, return the result string.

        Expected format: estimate(param1=val1, param2=val2, ...)
        Falls back to a simulated result if the API isn't reachable.
        """
        match = re.match(r"(\w+)\((.*)\)", tool_text.strip(), re.DOTALL)
        if not match:
            return json.dumps({"error": "Could not parse tool call"})

        func_name = match.group(1)
        args_str = match.group(2)

        if func_name == "estimate":
            return self._run_estimate(args_str)
        elif func_name == "rate_lookup":
            return self._run_rate_lookup(args_str)
        else:
            return json.dumps({"error": f"Unknown tool: {func_name}"})

    def _run_estimate(self, args_str: str) -> str:
        """Call the PyTorch cost estimator API or compute locally."""
        params = self._parse_kwargs(args_str)
        try:
            import urllib.request
            payload = json.dumps(params).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8001/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.read().decode()
        except Exception:
            # Fallback: simple heuristic estimate
            lf = float(params.get("linear_feet", 1000))
            size = int(params.get("pipe_diameter", 8))
            base = {4: 35, 6: 50, 8: 72, 10: 95, 12: 120, 16: 170, 24: 240, 36: 380}
            cost = lf * base.get(size, 80)
            return json.dumps({
                "estimated_cost": round(cost, 2),
                "cost_per_lf": round(cost / max(lf, 1), 2),
                "note": "heuristic fallback (API not available)"
            })

    def _run_rate_lookup(self, args_str: str) -> str:
        """Look up a rate from the JSON data files."""
        params = self._parse_kwargs(args_str)
        category = params.get("category", "materials")
        item = params.get("item", "")
        try:
            data_dir = SCRIPT_DIR / ".." / ".." / "data"
            with open(data_dir / f"{category}.json") as f:
                data = json.load(f)
            return json.dumps({"result": str(data), "query": item})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _parse_kwargs(self, s: str) -> dict:
        """Parse keyword arguments from a string like key1=val1, key2=val2."""
        result = {}
        for part in re.split(r",\s*(?=\w+=)", s):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                result[k] = v
        return result

    # ---- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        handle_tools: bool = True,
    ) -> str:
        """Generate text from a prompt, intercepting tool calls."""
        tokens = self.tokenizer.encode(prompt)

        # Truncate to fit context window
        if len(tokens) > self.cfg.block_size - max_new_tokens:
            tokens = tokens[-(self.cfg.block_size - max_new_tokens):]

        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated_ids = []

        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]  # last token logits

            # Temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                generated_ids.append(next_id.item())
                if next_id.item() == self.end_id:
                    break
                continue

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            generated_ids.append(next_id.item())

            # Check for end token
            if next_id.item() == self.end_id:
                break

            # Check for tool call token
            if handle_tools and next_id.item() == self.tool_call_id:
                # Continue generating to get the tool call content
                tool_tokens = []
                for _ in range(128):  # max tool call length
                    idx_cond = idx[:, -self.cfg.block_size:]
                    logits, _ = self.model(idx_cond)
                    logits = logits[:, -1, :] / max(temperature, 0.1)
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, -1:]] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    nid = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, nid], dim=1)
                    generated_ids.append(nid.item())
                    tool_tokens.append(nid.item())

                    # Stop at tool_result token or end token
                    if nid.item() in (self.tool_result_id, self.end_id):
                        break

                # Extract and execute tool call
                tool_text = self.tokenizer.decode(tool_tokens[:-1])  # exclude the stop token
                result = self.execute_tool(tool_text)

                # Inject tool result back into the context
                result_str = f"<|tool_result|>{result}"
                result_tokens = self.tokenizer.encode(result_str)
                result_tensor = torch.tensor([result_tokens], dtype=torch.long, device=self.device)
                idx = torch.cat([idx, result_tensor], dim=1)
                generated_ids.extend(result_tokens)

        output = self.tokenizer.decode(generated_ids)

        # Clean up: remove anything after <|end|>
        if "<|end|>" in output:
            output = output[:output.index("<|end|>")]

        # Remove tool call/result tokens for clean display
        clean = output
        clean = re.sub(r"<\|tool_call\|>.*?<\|tool_result\|>", "", clean)
        # But preserve the assistant's text after tool result
        if "<|tool_result|>" in output:
            parts = output.split("<|tool_result|>")
            if len(parts) > 1:
                # Get everything after the JSON result
                after_result = parts[-1]
                json_end = after_result.find("}")
                if json_end != -1:
                    clean = after_result[json_end + 1:].strip()
                else:
                    clean = after_result.strip()

        return clean.strip()

    def chat(self, conversation: list[dict]) -> str:
        """Generate a response given a conversation history.

        conversation: [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        prompt = ""
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|{role}|>{content}<|end|>\n"
        prompt += "<|assistant|>"

        return self.generate(prompt)


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    engine = MasonEngine(model_path=args.model)

    if args.interactive:
        print("\nMason Transformer (interactive mode)")
        print("Type 'quit' to exit, 'reset' to clear history\n")
        history = [{"role": "system", "content": "You are Mason, a personal AI assistant built by Mason Earl. You have deep knowledge of construction estimating, technology, health, and everything on masonearl.com. Use the estimate tool when users ask for project costs."}]

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() in ("reset", "clear"):
                history = [history[0]]
                print("History cleared.\n")
                continue
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})
            response = engine.chat(history)
            history.append({"role": "assistant", "content": response})
            print(f"Mason: {response}\n")

    elif args.prompt:
        full_prompt = (
            "<|system|>You are Mason, a personal AI assistant built by Mason Earl. You handle construction estimating, tech questions, and anything on masonearl.com. "
            "Use the estimate tool when users ask for project costs.<|end|>\n"
            f"<|user|>{args.prompt}<|end|>\n"
            "<|assistant|>"
        )
        response = engine.generate(
            full_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(response)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
