#!/usr/bin/env python3
"""Build the training corpus from structured domain data + full website content.

Reads JSON/CSV files and generates natural-language text in multiple categories:
  1. Knowledge passages  -- prose paragraphs about materials, labor, equipment, etc.
  2. Rate Q&A pairs       -- short question/answer about specific prices
  3. Synthetic conversations with tool calls -- realistic user/assistant exchanges
  4. Website content      -- extracted text from all HTML, MD, and JSON on masonearl.com
  5. Construction industry knowledge -- AGC, OSHA, RSMeans, Utah agencies, math
  6. General knowledge conversations -- broader Q&A for natural chat ability
"""

import json
import csv
import os
import re
import random
import itertools
from pathlib import Path
from html.parser import HTMLParser

SCRIPT_DIR = Path(__file__).resolve().parent

# Support both original repo structure (../../data) and contech1 flat structure (./data)
_data_orig = SCRIPT_DIR / ".." / ".." / "data"
_data_flat = SCRIPT_DIR / "data"
DATA_DIR = _data_orig if (_data_orig / "materials.json").exists() else _data_flat

# SITE_ROOT: only scrape website content if running inside the masonearl.com repo.
# In the flat contech1 repo, skip website scraping entirely to avoid scanning the filesystem.
_site_orig = SCRIPT_DIR / ".." / ".." / ".." / ".." / ".."
SITE_ROOT = _site_orig if (_site_orig / "pages").exists() else None

CORPUS_DIR = SCRIPT_DIR / "corpus"
CORPUS_DIR.mkdir(exist_ok=True)

random.seed(42)


# ---- HTML text extractor ---------------------------------------------------

class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract readable text."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "path", "meta", "link", "head"}

    def __init__(self):
        super().__init__()
        self._text = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag.lower() in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
            self._text.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._text.append(data)

    def get_text(self):
        raw = "".join(self._text)
        lines = [line.strip() for line in raw.splitlines()]
        lines = [l for l in lines if l]
        return "\n".join(lines)


def extract_html_text(filepath):
    """Extract readable text from an HTML file."""
    try:
        html = filepath.read_text(errors="ignore")
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else filepath.stem.replace("-", " ").title()
        return title, text
    except Exception:
        return filepath.stem, ""


def extract_md_text(filepath):
    """Extract text from a markdown file."""
    try:
        text = filepath.read_text(errors="ignore")
        title_match = re.match(r"^#\s+(.+)", text)
        title = title_match.group(1).strip() if title_match else filepath.stem.replace("-", " ").title()
        return title, text
    except Exception:
        return filepath.stem, ""


def extract_json_text(filepath):
    """Convert JSON data into readable text."""
    try:
        data = json.loads(filepath.read_text(errors="ignore"))
        title = filepath.stem.replace("-", " ").replace("_", " ").title()
        lines = [f"# {title}\n"]

        def _flatten(obj, depth=0):
            if depth > 4:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        lines.append(f"{'  ' * depth}{k}: {v}")
                    elif isinstance(v, list) and all(isinstance(i, (str, int, float)) for i in v):
                        lines.append(f"{'  ' * depth}{k}: {', '.join(str(i) for i in v)}")
                    else:
                        lines.append(f"{'  ' * depth}{k}:")
                        _flatten(v, depth + 1)
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:50]):
                    if isinstance(item, (str, int, float)):
                        lines.append(f"{'  ' * depth}- {item}")
                    elif isinstance(item, dict):
                        _flatten(item, depth)
                        lines.append("")

        _flatten(data)
        return title, "\n".join(lines)
    except Exception:
        return filepath.stem, ""


def build_website_corpus():
    """Scrape all website content into training text."""
    if SITE_ROOT is None:
        print("  (Website scraping skipped — not in masonearl.com repo)")
        return "", [], 0
    site = SITE_ROOT.resolve()
    sections = []
    page_count = 0

    skip_dirs = {".venv", "node_modules", ".git", "checkpoints", "corpus",
                 "__pycache__", ".cursor", "assets", "image"}
    skip_files = {"package.json", "vercel.json", "package-lock.json",
                  "tsconfig.json", "robots.txt"}

    for html_file in sorted(site.rglob("*.html")):
        if any(part in skip_dirs for part in html_file.parts):
            continue
        if html_file.name in skip_files:
            continue
        title, text = extract_html_text(html_file)
        if len(text) > 50:
            rel_path = html_file.relative_to(site)
            sections.append(f"# {title}\nSource: {rel_path}\n\n{text}")
            page_count += 1

    for md_file in sorted(site.rglob("*.md")):
        if any(part in skip_dirs for part in md_file.parts):
            continue
        title, text = extract_md_text(md_file)
        if len(text) > 30:
            rel_path = md_file.relative_to(site)
            sections.append(f"# {title}\nSource: {rel_path}\n\n{text}")
            page_count += 1

    for json_file in sorted(site.rglob("*.json")):
        if any(part in skip_dirs for part in json_file.parts):
            continue
        if json_file.name in skip_files:
            continue
        if json_file.name in ("tokenizer.json",):
            continue
        title, text = extract_json_text(json_file)
        if len(text) > 50:
            rel_path = json_file.relative_to(site)
            sections.append(f"# {title}\nSource: {rel_path}\n\n{text}")
            page_count += 1

    qa_pairs = _generate_website_qa(sections)
    full_text = "\n\n---\n\n".join(sections)
    return full_text, qa_pairs, page_count


def _generate_website_qa(sections):
    """Generate Q&A training pairs from website section titles."""
    pairs = []
    for section in sections:
        lines = section.split("\n")
        if not lines:
            continue
        title = lines[0].lstrip("# ").strip()
        source = lines[1].replace("Source: ", "").strip() if len(lines) > 1 else ""
        content_preview = " ".join(lines[3:8]).strip()[:300]

        if not content_preview or len(content_preview) < 20:
            continue

        q_templates = [
            f"What is on the {title} page?",
            f"Tell me about {title}.",
            f"What do you know about {title}?",
        ]
        a_template = f"The {title} page ({source}) covers: {content_preview}"
        pairs.append((random.choice(q_templates), a_template))

    return pairs


# ---- helpers ---------------------------------------------------------------

def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)

def fmt(n):
    if n == int(n):
        return f"{int(n):,}"
    return f"{n:,.2f}"

def pretty_size(s):
    return s.replace("_", "-").replace("inch", '"')

def pretty_pipe_type(t):
    return {"pvc": "PVC", "dip": "DIP (ductile iron)", "hdpe": "HDPE",
            "corrugated": "corrugated"}.get(t, t.upper())


# ---- 1. knowledge passages ------------------------------------------------

def build_knowledge(materials, labor, equipment, production, terms):
    lines = []

    lines.append("# Construction Glossary\n")
    for category, items in terms.items():
        cat_nice = category.replace("_", " ").title()
        lines.append(f"## {cat_nice}\n")
        for key, desc in items.items():
            lines.append(f"{key}: {desc}\n")
        lines.append("")

    lines.append("# Material Pricing\n")
    for pipe_type, sizes in materials.get("pipe", {}).items():
        ptype = pretty_pipe_type(pipe_type)
        lines.append(f"## {ptype} Pipe\n")
        for size, info in sizes.items():
            sz = pretty_size(size)
            lines.append(
                f"{ptype} pipe in {sz} diameter costs ${fmt(info['cost'])} per "
                f"linear foot. Include a {int(info['waste_factor']*100)}% waste factor "
                f"when estimating quantities.\n"
            )
        lines.append("")

    for mat_category in ["concrete", "rebar", "aggregate", "asphalt"]:
        if mat_category in materials:
            lines.append(f"## {mat_category.title()}\n")
            for item, info in materials[mat_category].items():
                name = item.replace("_", " ").title()
                lines.append(
                    f"{name} costs ${fmt(info['cost'])} per {info['unit']}. "
                    f"Waste factor: {int(info.get('waste_factor',0)*100)}%.\n"
                )
            lines.append("")

    lines.append("## Fittings & Structures\n")
    for item, info in materials.get("fittings", {}).items():
        name = item.replace("_", " ").title()
        lines.append(f"{name}: ${fmt(info['cost'])} each.\n")

    lines.append("\n## Miscellaneous Materials\n")
    for item, info in materials.get("misc", {}).items():
        name = item.replace("_", " ").title()
        lines.append(f"{name}: ${fmt(info['cost'])} per {info['unit']}.\n")

    lines.append("\n# Labor Rates\n")
    for role, info in labor.get("rates", {}).items():
        name = role.replace("_", " ").title()
        lines.append(
            f"A {name} ({info['description']}) has a base hourly rate of "
            f"${fmt(info['base_hourly'])}. With a burden rate of "
            f"{info['burden_rate']}x, the fully burdened cost is "
            f"${fmt(info['burdened_hourly'])}/hr. Overtime is "
            f"{info['overtime_multiplier']}x base.\n"
        )

    lines.append("\n## Typical Crew Compositions\n")
    for crew_key, crew in labor.get("typical_crews", {}).items():
        members_str = ", ".join(
            f"{m['count']} {m['role'].replace('_',' ')}" for m in crew["members"]
        )
        lines.append(
            f"{crew['description']}: {members_str}. "
            f"Daily crew cost: ${fmt(crew['daily_cost'])}.\n"
        )

    lines.append("\n# Equipment Rates\n")
    for cat, items in equipment.items():
        if not isinstance(items, dict): continue  # skip "notes" string key
        cat_nice = cat.replace("_", " ").title()
        lines.append(f"## {cat_nice}\n")
        for _, info in items.items():
            if not isinstance(info, dict): continue
            lines.append(
                f"{info['description']}: ${fmt(info['daily_rate'])}/day, "
                f"${fmt(info['weekly_rate'])}/week, "
                f"${fmt(info['monthly_rate'])}/month. "
                f"Fuel cost: ${fmt(info['fuel_per_hour'])}/hr. "
                f"Typical use: {info['typical_use']}.\n"
            )
        lines.append("")

    lines.append("# Production Rates\n")
    for proj_type, sizes in production.get("pipe_installation", {}).items():
        ptype = proj_type.replace("_", " ").title()
        lines.append(f"## {ptype} Installation\n")
        for size, soils in sizes.items():
            sz = pretty_size(size)
            for soil, info in soils.items():
                lines.append(
                    f"Installing {sz} {ptype.lower()} in {soil} soil: "
                    f"{info['lf_per_day']} LF/day. "
                    f"Typical depth range: {info['depth_range']}.\n"
                )
        lines.append("")

    lines.append("## Condition Multipliers\n")
    for cond, mult in production.get("condition_multipliers", {}).items():
        lines.append(
            f"{cond.replace('_',' ').title()}: multiply production by {mult} "
            f"({int((1-mult)*100)}% slower).\n"
        )

    lines.append("\n## Season Multipliers\n")
    for season, mult in production.get("season_multipliers", {}).items():
        lines.append(f"{season.title()}: {mult}x production rate.\n")

    lines.append("\n## Location Multipliers (Utah)\n")
    for loc, mult in production.get("location_multipliers", {}).items():
        lines.append(f"{loc.replace('_',' ').title()}: {mult}x.\n")

    return "\n".join(lines)


# ---- 2. rate Q&A pairs ---------------------------------------------------

def build_rate_qa(materials, labor, equipment):
    pairs = []

    for pipe_type, sizes in materials.get("pipe", {}).items():
        ptype = pretty_pipe_type(pipe_type)
        for size, info in sizes.items():
            sz = pretty_size(size)
            q_templates = [
                f"How much does {sz} {ptype} pipe cost?",
                f"What is the price per foot for {sz} {ptype}?",
                f"Give me the unit cost for {sz} {ptype} pipe.",
                f"What's the material cost of {sz} {ptype}?",
            ]
            a_templates = [
                f"{sz} {ptype} pipe costs ${fmt(info['cost'])} per linear foot, with a {int(info['waste_factor']*100)}% waste factor.",
                f"The unit price for {sz} {ptype} is ${fmt(info['cost'])}/LF. Remember to add {int(info['waste_factor']*100)}% for waste.",
            ]
            pairs.append((random.choice(q_templates), random.choice(a_templates)))

    for role, info in labor.get("rates", {}).items():
        name = role.replace("_", " ").title()
        q_options = [
            f"What does a {name.lower()} cost per hour?",
            f"How much do I pay for a {name.lower()}?",
            f"What is the burdened rate for a {name.lower()}?",
        ]
        a_options = [
            f"A {name.lower()} costs ${fmt(info['burdened_hourly'])}/hr fully burdened (base: ${fmt(info['base_hourly'])}/hr, burden rate {info['burden_rate']}x).",
            f"The burdened hourly rate for a {name.lower()} is ${fmt(info['burdened_hourly'])}.",
        ]
        pairs.append((random.choice(q_options), random.choice(a_options)))

    for cat, items in equipment.items():
        if not isinstance(items, dict): continue
        for _, info in items.items():
            if not isinstance(info, dict): continue
            desc = info.get("description", cat)
            q_options = [
                f"How much does a {desc.lower()} cost per day?",
                f"What's the rental rate for a {desc.lower()}?",
            ]
            a_options = [
                f"A {desc.lower()} runs ${fmt(info['daily_rate'])}/day, ${fmt(info['weekly_rate'])}/week, or ${fmt(info['monthly_rate'])}/month. Fuel is ${fmt(info['fuel_per_hour'])}/hr.",
                f"The daily rate for a {desc.lower()} is ${fmt(info['daily_rate'])}. Weekly: ${fmt(info['weekly_rate'])}. Monthly: ${fmt(info['monthly_rate'])}.",
            ]
            pairs.append((random.choice(q_options), random.choice(a_options)))

    return pairs


# ---- 3. synthetic conversations with tool calls ---------------------------

PROJECT_TYPES = ["waterline", "sewer", "storm_drain", "gas", "electrical"]
SOIL_TYPES = ["sand", "clay", "gravel", "rock"]
PIPE_SIZES = [4, 6, 8, 10, 12, 16, 24, 36]
DEPTHS = [3, 4, 5, 6, 8, 10, 12, 15]

def generate_conversations(n=600):
    convos = []

    for _ in range(n):
        proj = random.choice(PROJECT_TYPES)
        soil = random.choice(SOIL_TYPES)
        size = random.choice(PIPE_SIZES)
        lf = random.choice(range(100, 10001, 50))
        depth = random.choice(DEPTHS)
        dewater = random.choice([0, 1])
        road_cross = random.randint(0, 5)
        season = random.choice(["winter", "spring", "summer", "fall"])
        location = random.choice(["salt_lake_metro", "utah_county", "davis_weber",
                                   "rural_utah", "mountain_areas"])
        crew_size = random.randint(4, 8)
        manholes = random.randint(0, int(lf / 300) + 1)
        markup = round(random.uniform(0.08, 0.20), 2)
        proj_pretty = proj.replace("_", " ")

        user_templates = [
            f"Estimate {lf} LF of {size}-inch {proj_pretty} in {soil} soil at {depth} ft deep.",
            f"What would {lf} linear feet of {size}\" {proj_pretty} cost in {soil}?",
            f"I need a price for {lf} LF of {size}-inch {proj_pretty}, {soil} soil, {depth} feet deep.",
            f"Can you estimate {lf} feet of {size}-inch {proj_pretty}? Soil is {soil}, depth around {depth} ft.",
            f"How much for {lf} LF {size}\" {proj_pretty}? {soil.title()} soil, approximately {depth}' deep.",
            f"Give me a ballpark on {lf} LF of {size}-inch {proj_pretty} in {soil} conditions.",
            f"Price out {lf} linear feet of {size}-inch {proj_pretty} for me. {soil.title()} soil, {depth} ft depth.",
        ]

        user_msg = random.choice(user_templates)

        tool_call = (
            f"<|tool_call|>estimate("
            f"project_type=\"{proj}\", "
            f"linear_feet={lf}, "
            f"pipe_diameter={size}, "
            f"depth={depth}, "
            f"soil_type=\"{soil}\", "
            f"dewatering={dewater}, "
            f"road_crossings={road_cross}, "
            f"season=\"{season}\", "
            f"location=\"{location}\", "
            f"crew_size={crew_size}, "
            f"manholes={manholes}, "
            f"markup={markup})"
        )

        base_per_lf = {4: 35, 6: 50, 8: 72, 10: 95, 12: 120, 16: 170, 24: 240, 36: 380}
        soil_mult = {"sand": 1.0, "clay": 1.2, "gravel": 1.1, "rock": 2.5}
        cost_est = lf * base_per_lf.get(size, 80) * soil_mult.get(soil, 1.0)
        cost_est *= (1 + markup)
        cost_est = round(cost_est, 2)

        tool_result = f"<|tool_result|>{{\"estimated_cost\": {cost_est}, \"cost_per_lf\": {round(cost_est/lf, 2)}, \"duration_days\": {max(1, lf // random.randint(80, 180))}}}"

        duration = max(1, lf // random.randint(80, 180))
        response_templates = [
            (
                f"Based on the estimator, {lf} LF of {size}-inch {proj_pretty} in {soil} soil "
                f"comes to approximately ${fmt(cost_est)}. That's about ${fmt(round(cost_est/lf, 2))}/LF. "
                f"Estimated duration: {duration} working days."
            ),
            (
                f"The estimate for {lf} linear feet of {size}\"{proj_pretty} is ${fmt(cost_est)} total "
                f"(${fmt(round(cost_est/lf, 2))} per foot). This includes a {int(markup*100)}% markup. "
                f"Expect roughly {duration} days of production."
            ),
            (
                f"I ran the numbers: ${fmt(cost_est)} for {lf} LF of {size}-inch {proj_pretty} "
                f"in {soil}. That breaks down to ${fmt(round(cost_est/lf, 2))} per linear foot, "
                f"with an estimated {duration}-day schedule."
            ),
        ]

        assistant_msg = random.choice(response_templates)

        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. You handle construction estimating, technology, health, and anything on masonearl.com. "
            f"Use the estimate tool when users ask for project costs.<|end|>\n"
            f"<|user|>{user_msg}<|end|>\n"
            f"<|assistant|>{tool_call}{tool_result}\n{assistant_msg}<|end|>"
        )
        convos.append(convo)

    # ---- general Q&A conversations (no tool call) ----
    general_pairs = [
        ("What is a takeoff?",
         "A takeoff is the process of measuring quantities from construction plans and blueprints. You go through each sheet and count/measure all the items needed for the project."),
        ("What does dewatering mean?",
         "Dewatering is the process of pumping groundwater out of a trench to allow work in dry conditions. It's common when trenching below the water table."),
        ("What is compaction?",
         "Compaction is the process of mechanically compacting soil or fill material to a required density, typically 95% Proctor. It's done in lifts after backfilling a trench."),
        ("What is a thrust block?",
         "A thrust block is a concrete block poured behind pipe fittings to resist pressure forces at bends, tees, and dead ends in pressurized waterlines."),
        ("When do I need shoring?",
         "OSHA requires shoring or a trench box for excavations deeper than 5 feet. Shoring prevents trench walls from caving in and is critical for worker safety."),
        ("What's the difference between open cut and bore?",
         "Open cut means digging a trench from the surface to install pipe. A bore drills horizontally under a road or obstruction without disturbing the surface. Bores are more expensive but avoid road closures."),
        ("What pipe material should I use for water?",
         "For waterlines, ductile iron pipe (DIP) is common for larger diameters (8\"+) due to its strength. PVC is typical for smaller sizes. HDPE is used where flexibility is needed."),
        ("Tell me about prevailing wage.",
         "Prevailing wage refers to government-mandated minimum wage rates for public works projects, set by Davis-Bacon. Workers on federally funded projects must be paid at least these rates."),
        ("What is retainage?",
         "Retainage is a percentage of payment (typically 5-10%) withheld by the owner until the project is substantially complete. It's meant to ensure the contractor finishes the work."),
        ("What does mobilization include?",
         "Mobilization covers moving equipment, materials, and crew to the job site at the start of the project. It includes transportation, setup of temporary facilities, and initial site preparation."),
        ("How do condition multipliers work?",
         "Condition multipliers adjust production rates for difficult conditions. For example, dewatering multiplies production by 0.70 (30% slower), and rock excavation by 0.35 (65% slower)."),
        ("What crew do I need for an 8-inch waterline?",
         "A typical 8-16 inch waterline crew includes 1 foreman, 2 operators, 2 pipe layers, and 3 laborers. The daily crew cost is around $3,408."),
        ("What does a foreman make?",
         "A foreman has a base hourly rate of $35.00. With a 1.45x burden rate, the fully burdened cost is $50.75/hr. Overtime is 1.5x the base rate."),
        ("How fast can we install 12-inch sewer in clay?",
         "For 12-inch sewer in clay soil, expect about 70 LF per day at 8-12 ft depth. That's with a standard sewer crew."),
        ("What equipment do I need for trenching?",
         "Standard trenching requires a mid-size excavator ($400/day), plate compactor ($75/day), and a dump truck ($350-500/day). For deep work, add a trench box ($125/day). In rock, add a hydraulic hammer ($350/day)."),
        ("How much does a fire hydrant cost?",
         "A fire hydrant assembly costs about $3,200 each. That's material only -- you'll also need the connection fittings and labor for installation."),
        ("What's the typical depth for waterline?",
         "Waterline depth varies by pipe size. 4-inch lines run 3-4 ft deep, while 12-inch lines go 4.5-6 ft deep. You need to stay below the frost line and meet local code requirements."),
        ("What is bedding material?",
         "Bedding is select material (usually pipe bedding aggregate at ~$20/ton) placed in the bottom of the trench for the pipe to rest on. It provides uniform support and protects the pipe from point loads."),
        ("What are submittals?",
         "Submittals are product data, shop drawings, or material samples sent to the engineer for approval before you install them. They verify you're using the right products per the specifications."),
        ("How does winter affect production?",
         "Winter conditions multiply production rates by 0.85 (15% slower). Frozen ground is harder to excavate, concrete curing is slower, and shorter daylight hours reduce productive time."),
        ("Hello",
         "Hey! I'm Mason, a personal AI trained on masonearl.com. I can estimate construction costs, answer tech questions, look up rates, and talk about anything on the site. What are you working on?"),
        ("What can you do?",
         "I can estimate construction project costs using a neural network model, look up current material prices, labor rates, and equipment costs, calculate production rates for pipe installation, and answer general construction questions. Just tell me about your project."),
        ("Thanks for your help",
         "You're welcome! Feel free to come back anytime you need an estimate or have questions about a project."),
        ("How does your estimating model work?",
         "I use a neural network trained on construction project data. You give me parameters like pipe size, footage, soil type, and depth, and the model predicts the total cost based on patterns it learned from previous projects. The more data we feed it, the better it gets."),
    ]

    for q, a in general_pairs:
        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>"
        )
        convos.append(convo)

    # ---- bid request / project start conversations ----
    bid_openers = [
        ("Can you help me bid a job?",
         "Sure, what kind of job? Give me the basics -- pipe type, size, and how much footage you're looking at."),
        ("I need to bid a project.",
         "Absolutely. What type of work is it? Waterline, sewer, storm drain, gas? And do you have footage and pipe size yet?"),
        ("Help me put together a bid.",
         "On it. Walk me through the project -- what kind of pipe, what diameter, and how many feet? I'll start building the numbers from there."),
        ("I have a job to price out.",
         "Let's do it. What are we looking at -- type of work and pipe size first, then we'll get into footage and conditions."),
        ("Can you estimate something for me?",
         "Yeah, what do you have? Tell me about the project -- pipe type, size, footage, and soil conditions are the big ones."),
        ("I need a number on a project.",
         "Sure thing. What kind of project? Give me the pipe type and size to start, and we'll build out from there."),
        ("We're bidding a job next week.",
         "Let's get ahead of it. What type of work -- sewer, water, gas? And do you have plans with footage and pipe sizes yet?"),
        ("Got a project I need help with.",
         "What are we working with? Type of pipe, size, and footage are the starting point. I'll pull the rates and build the estimate as we go."),
        ("I'm working on a bid.",
         "Nice, what's the scope? Give me the basics -- what kind of pipe, what size, how many feet -- and I'll start running numbers."),
        ("Can you help me price some work?",
         "Absolutely. What kind of work are we pricing? Pipe type, diameter, and linear feet are the key inputs I need to get started."),
        ("I need to bid a waterline job.",
         "Waterline -- got it. What size pipe and how many linear feet? And do you know the soil type at the site?"),
        ("Help me estimate a sewer project.",
         "Sure, sewer. What diameter pipe and how many feet are we talking? I'll also need soil type and depth when you have them."),
        ("I'm pricing out a gas line.",
         "Gas line, nice. What size -- 2-inch, 4-inch? And how much footage? Those are the two biggest cost drivers."),
        ("We need to bid some storm drain work.",
         "Storm drain. What pipe size and how many LF? Also, is it concrete pipe, corrugated, or HDPE?"),
        ("I have an electrical duct bank to price.",
         "Electrical duct bank -- what size and how many linear feet? And do you know how many conduits in the bank?"),
        ("I need to bid a 2-inch gas line for a client.",
         "2-inch gas, got it. How many linear feet? And what's the soil like on site -- sand, clay, gravel, or rock?"),
        ("We're bidding 8-inch waterline for the city.",
         "8-inch waterline for a city project. How many feet? Is this prevailing wage? And do you know the soil conditions?"),
        ("Client wants a price on some sewer replacement.",
         "Sewer replacement -- what diameter are we replacing, and how many feet? Knowing the depth and soil type will tighten up the estimate."),
        ("Need to price 12-inch DIP for a developer.",
         "12-inch DIP waterline. How many linear feet for the developer? And what are site conditions -- soil type, depth, any rock?"),
        ("I have a question.",
         "Go for it. What do you need help with?"),
        ("Can I ask you something?",
         "Of course. What's on your mind?"),
        ("Hey, I need your help.",
         "What's up? Tell me what you're working on and I'll see what I can do."),
        ("I could use some help.",
         "Sure, what do you need? Project estimate, rate lookup, or something else?"),
    ]

    for q, a in bid_openers:
        convo = (
            f"<|system|>You are Mason, a personal AI assistant. When users ask vague questions, ask clarifying questions to understand the project before estimating.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>"
        )
        convos.append(convo)
        convo2 = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. Use the estimate tool when users ask for project costs.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>"
        )
        convos.append(convo2)

    # ---- multi-turn conversations (expanded) ----
    multi_turns = [
        [("user", "I need to estimate a sewer project."),
         ("assistant", "Sure, I can help with a sewer estimate. How many linear feet of pipe do you need?"),
         ("user", "About 3,000 LF of 8-inch."),
         ("assistant", "Got it -- 3,000 LF of 8-inch sewer. What's the soil type at the site?"),
         ("user", "Mostly clay with some rock."),
         ("assistant", "Clay with rock is going to slow things down. What depth are we looking at?"),
         ("user", "8 to 10 feet."),
         ("assistant", "<|tool_call|>estimate(project_type=\"sewer\", linear_feet=3000, pipe_diameter=8, depth=9, soil_type=\"clay\", dewatering=0, road_crossings=0, season=\"summer\", location=\"salt_lake_metro\", crew_size=7, manholes=10, markup=0.12)<|tool_result|>{\"estimated_cost\": 291600.00, \"cost_per_lf\": 97.20, \"duration_days\": 33}\nBased on 3,000 LF of 8-inch sewer in clay at ~9 ft depth, the estimate is approximately $291,600 ($97.20/LF). With a standard sewer crew, expect about 33 working days. The clay soil and depth will require shoring.")],
        [("user", "Can you help me bid a job?"),
         ("assistant", "Sure, what kind of job? Give me the basics -- pipe type, size, and how much footage you're looking at."),
         ("user", "2-inch gas line, about 400 feet for a residential project."),
         ("assistant", "2-inch gas, 400 LF, residential. What's the soil -- sandy, clay? And how deep do we need to go?"),
         ("user", "Sand, about 3 feet deep."),
         ("assistant", "<|tool_call|>estimate(project_type=\"gas\", linear_feet=400, pipe_diameter=2, depth=3, soil_type=\"sand\", dewatering=0, road_crossings=0, season=\"summer\", location=\"salt_lake_metro\", crew_size=4, manholes=0, markup=0.15)<|tool_result|>{\"estimated_cost\": 16100.00, \"cost_per_lf\": 40.25, \"duration_days\": 4}\nFor 400 LF of 2-inch gas in sand at 3 ft deep, you're looking at about $16,100 ($40.25/LF). That's about 4 working days with a standard crew. Sand is the easiest soil to work in, so production should be smooth.")],
        [("user", "I'm pricing out a waterline."),
         ("assistant", "Waterline -- what size pipe and how much footage?"),
         ("user", "12-inch, about 5,000 feet."),
         ("assistant", "That's a good-sized project. What's the soil like on site?"),
         ("user", "Clay."),
         ("assistant", "Clay soil, got it. What about depth -- are we looking at standard 4-5 ft or deeper?"),
         ("user", "6 feet."),
         ("assistant", "<|tool_call|>estimate(project_type=\"waterline\", linear_feet=5000, pipe_diameter=12, depth=6, soil_type=\"clay\", dewatering=0, road_crossings=2, season=\"summer\", location=\"salt_lake_metro\", crew_size=6, manholes=0, markup=0.12)<|tool_result|>{\"estimated_cost\": 806400.00, \"cost_per_lf\": 161.28, \"duration_days\": 56}\nFor 5,000 LF of 12-inch waterline in clay at 6 ft deep, the estimate comes to approximately $806,400 ($161.28/LF). Expect around 56 working days. At 6 ft depth you'll need shoring per OSHA requirements.")],
        [("user", "What size excavator do I need for a 6-foot deep waterline?"),
         ("assistant", "For a 6-foot deep waterline, a mid-size excavator (15-20 ton) is your best bet. It runs $400/day and handles standard utility trenching well. You won't need the large excavator unless you hit rock."),
         ("user", "And what about compaction?"),
         ("assistant", "For trench backfill at that depth, a vibratory plate compactor ($75/day) is standard. You'll compact in 8-12 inch lifts. For the road base or larger areas, step up to a walk-behind roller ($100/day) or ride-on roller ($300/day).")],
        [("user", "Compare PVC and DIP for a 12-inch waterline."),
         ("assistant", "For 12-inch waterline:\n\n- PVC: $22.00/LF material cost, lighter, easier to install, good for standard conditions\n- DIP (Ductile Iron): $32.00/LF material cost, stronger, better for high-pressure or challenging conditions\n\nDIP costs about 45% more in material but is more durable. For most standard waterline projects in normal soil, PVC is the cost-effective choice. DIP is worth the premium for high-traffic areas or where you need extra strength.")],
        [("user", "I need to put together a bid for Highland Drive waterline replacement."),
         ("assistant", "Highland Drive waterline replacement -- what size pipe are they specifying? And do you have the total footage from the plans?"),
         ("user", "8-inch DIP, about 2,200 feet. There are 6 fire hydrants too."),
         ("assistant", "8-inch DIP, 2,200 LF with 6 hydrants. What's the soil type and depth? And are there road crossings?"),
         ("user", "It's mostly clay, 5 feet deep, and there are 3 road crossings."),
         ("assistant", "<|tool_call|>estimate(project_type=\"waterline\", linear_feet=2200, pipe_diameter=8, depth=5, soil_type=\"clay\", dewatering=0, road_crossings=3, season=\"summer\", location=\"salt_lake_metro\", crew_size=6, manholes=0, markup=0.12)<|tool_result|>{\"estimated_cost\": 211344.00, \"cost_per_lf\": 96.07, \"duration_days\": 24}\nFor 2,200 LF of 8-inch DIP waterline in clay at 5 ft deep with 3 road crossings, the pipe and install estimate is approximately $211,344 ($96.07/LF). Add 6 fire hydrant assemblies at ~$3,200 each ($19,200) and you're at roughly $230,544 total. Expect about 24 working days of production.")],
        [("user", "I don't really know where to start, I have plans for a project."),
         ("assistant", "No problem. What kind of project is it -- waterline, sewer, storm drain, gas? And what do the plans show for pipe size and footage?"),
         ("user", "It's a sewer line, looks like 8-inch PVC, maybe 1,000 feet."),
         ("assistant", "8-inch PVC sewer, about 1,000 LF -- good start. Do you see depth callouts on the plans? And what about soil -- is there a geotech report?"),
         ("user", "The profiles show about 10 feet deep. No geotech that I can see."),
         ("assistant", "10 feet deep is getting into shoring territory. Without a geotech, I'd assume clay for the area. Want me to run a preliminary estimate with those numbers?"),
         ("user", "Yeah, let's see what it looks like."),
         ("assistant", "<|tool_call|>estimate(project_type=\"sewer\", linear_feet=1000, pipe_diameter=8, depth=10, soil_type=\"clay\", dewatering=0, road_crossings=0, season=\"summer\", location=\"salt_lake_metro\", crew_size=6, manholes=3, markup=0.12)<|tool_result|>{\"estimated_cost\": 96768.00, \"cost_per_lf\": 96.77, \"duration_days\": 12}\nPreliminary estimate for 1,000 LF of 8-inch sewer in clay at 10 ft deep: about $96,768 ($96.77/LF). That includes shoring costs for the depth. Roughly 12 working days. Once you get a geotech or more details from the plans, we can tighten this up.")],
        [("user", "How much do I mark up a bid?"),
         ("assistant", "Standard markup ranges from 8% to 20% depending on the project. Public works bids tend to run tighter (8-12%) since they're competitive. Private work or design-build gives you more room (12-20%). It depends on your overhead, risk tolerance, and how badly you want the job.")],
        [("user", "That estimate seems high, can we bring it down?"),
         ("assistant", "A few ways to bring the number down: reduce markup, use a smaller crew (saves on daily labor but extends the schedule), or see if PVC is spec'd instead of DIP. Soil type is the biggest cost driver -- if the actual conditions are better than what we assumed, the number drops. What part seems high to you?")],
        [("user", "Hey"),
         ("assistant", "Hey! What are you working on?"),
         ("user", "Got a gas line to price."),
         ("assistant", "Gas line -- what size and how many feet?"),
         ("user", "4-inch, 800 feet, gravel soil."),
         ("assistant", "<|tool_call|>estimate(project_type=\"gas\", linear_feet=800, pipe_diameter=4, depth=4, soil_type=\"gravel\", dewatering=0, road_crossings=1, season=\"summer\", location=\"salt_lake_metro\", crew_size=5, manholes=0, markup=0.12)<|tool_result|>{\"estimated_cost\": 35168.00, \"cost_per_lf\": 43.96, \"duration_days\": 8}\n800 LF of 4-inch gas in gravel comes to about $35,168 ($43.96/LF). Gravel is pretty forgiving soil so production should be solid -- roughly 8 working days.")],
    ]

    for turns in multi_turns:
        parts = ["<|system|>You are Mason, a personal AI assistant built by Mason Earl. You handle construction estimating, tech, health, and anything on masonearl.com. Use the estimate tool when users ask for project costs. When users are vague, ask clarifying questions first.<|end|>"]
        for role, text in turns:
            parts.append(f"<|{role}|>{text}<|end|>")
        convos.append("\n".join(parts))

    return convos


# ---- 4. construction industry knowledge (AGC, OSHA, RSMeans, Utah) --------

def build_construction_knowledge():
    """Generate comprehensive construction industry content for training."""
    sections = []

    # AGC and general contracting
    sections.append("""# Associated General Contractors (AGC) Overview

The Associated General Contractors of America (AGC) is the largest and oldest national construction trade association. AGC represents more than 27,000 firms including general contractors, specialty contractors, service providers, and suppliers. The AGC of Utah chapter serves contractors across the Wasatch Front and statewide.

## Bid Types and Delivery Methods

Hard Bid (Design-Bid-Build): The traditional delivery method. The owner hires a designer to create complete plans and specs, then puts the project out for competitive bidding. The lowest responsible bidder typically wins. This is the most common method for public works projects.

Design-Build: A single entity handles both design and construction. The owner contracts with one firm that manages the architect/engineer and the builder. This can reduce project timelines by overlapping design and construction phases.

CM/GC (Construction Manager / General Contractor): The owner hires a construction manager during the design phase to provide input on constructability, scheduling, and cost. The CM/GC then negotiates a Guaranteed Maximum Price (GMP) and manages construction. Common in Utah for UDOT and UTA projects.

Best Value: The owner evaluates proposals based on qualifications, approach, schedule, and price. Not purely lowest price. Used increasingly on public projects where quality and experience matter as much as cost.

Invitation to Bid (ITB): A formal solicitation to pre-qualified contractors. Common for public projects. Must follow state procurement rules.

Request for Proposal (RFP): Solicitation that evaluates qualifications and approach alongside price. Used for design-build and CM/GC projects.

Request for Qualifications (RFQ): First step in a two-phase selection. Shortlists firms based on experience, then solicits proposals from the shortlist.

## Bonding and Insurance

Bid Bond: Guarantees the contractor will honor their bid if selected. Typically 5% of the bid amount. Required on most public works projects.

Performance Bond: Guarantees the contractor will complete the work per the contract. Usually 100% of contract value. The surety company will pay to finish the project if the contractor defaults.

Payment Bond: Guarantees the contractor will pay subcontractors, suppliers, and laborers. Protects the owner from mechanic's liens. Usually 100% of contract value.

General Liability Insurance: Covers property damage and bodily injury on the job site. Minimum $1M per occurrence, $2M aggregate is standard.

Workers' Compensation: Required by Utah law. Covers medical expenses and lost wages for workers injured on the job. Rates vary by trade classification.

Builder's Risk Insurance: Covers damage to the structure under construction from fire, weather, theft, etc. Usually required by the owner.""")

    sections.append("""# OSHA Construction Safety Standards

The Occupational Safety and Health Administration (OSHA) sets and enforces safety standards for the construction industry under 29 CFR 1926.

## Excavation and Trenching (29 CFR 1926.650-652)

Trenches 5 feet deep or more require protective systems: sloping, benching, shoring, or a trench box (shield). Trenches 20 feet deep or more require a professional engineer to design the protective system.

Soil Classification: OSHA classifies soil as Type A (most stable - cohesive, uncracked clay), Type B (medium - crusite gravel, silt), or Type C (least stable - granular, submerged, sandy). The classification determines the allowable slope angle.

Maximum Allowable Slopes: Type A: 3/4:1 (53 degrees). Type B: 1:1 (45 degrees). Type C: 1.5:1 (34 degrees).

Competent Person: A designated competent person must inspect the excavation daily and after every rainstorm. They must have the authority to remove workers from the trench.

Access/Egress: Ladders, steps, or ramps must be provided in trenches 4 feet deep or more. Access points must be within 25 feet of all workers.

Spoil Placement: Excavated material (spoil) must be placed at least 2 feet from the edge of the trench.

Underground Utilities: All underground utilities must be located (call 811 / Blue Stakes in Utah) before excavation begins. Hand-dig within 24 inches of marked utilities.

Water Accumulation: Water must not be allowed to accumulate in trenches where workers are present. Dewatering equipment must be monitored by the competent person.

## Fall Protection (29 CFR 1926.500-503)

Fall protection is required at heights of 6 feet or more in construction. Methods include guardrail systems, safety net systems, and personal fall arrest systems.

## Scaffolding (29 CFR 1926.450-454)

Scaffolds must support 4 times the maximum intended load. Platforms must be at least 18 inches wide. Guardrails are required on scaffolds 10 feet or higher.

## Personal Protective Equipment

Hard hats, safety glasses, high-visibility vests, steel-toed boots, and hearing protection are required on most construction sites. Additional PPE depends on the specific hazard.""")

    sections.append("""# Construction Project Lifecycle

## 1. Preconstruction Phase

Plan Review: Obtain and review construction drawings, specifications, geotechnical reports, and addenda. Identify scope, quantities, and potential issues.

Quantity Takeoff: Measure all quantities from the plans. For utility work: linear feet of pipe by size and material, number of fittings, manholes, hydrants, valves, and other structures. For earthwork: cut/fill volumes, trench volumes, and paving areas.

Estimating: Apply unit costs to quantities. Include materials, labor, equipment, subcontractors, overhead, and profit. Adjust for site conditions, season, and location.

Bid Preparation: Compile the estimate into a bid form. Include required documents: bid bond, subcontractor list, non-collusion affidavit, and any required certifications.

## 2. Award and Contract Phase

Notice of Award: Owner notifies the winning bidder. Contractor has a specified time (usually 10-15 days) to execute the contract.

Contract Execution: Sign the contract, provide performance and payment bonds, and submit certificates of insurance.

Notice to Proceed (NTP): Owner issues NTP, which starts the contract time. The substantial completion date is typically calculated from the NTP date.

Preconstruction Meeting: Meet with the owner, engineer, and inspectors. Review the schedule, submittals, traffic control, environmental requirements, and communication protocols.

## 3. Construction Phase

Submittals: Submit product data, shop drawings, and material samples for engineer approval. This can take 2-4 weeks.

Scheduling: Develop and maintain a construction schedule. CPM (Critical Path Method) schedules are standard for larger projects.

Daily Operations: Manage crews, equipment, and materials. Complete daily reports documenting work performed, weather, equipment hours, labor hours, and any issues.

Quality Control: Perform required testing (compaction, concrete, welding). Document and correct any deficiencies.

Progress Payments: Submit monthly pay applications based on work completed. The engineer reviews and approves or modifies.

Change Orders: Document and negotiate changes in scope, conditions, or design. Get written approval before proceeding with extra work.

## 4. Project Closeout

Punch List: Walk the project with the owner and engineer. Document any remaining items or deficiencies.

Substantial Completion: The point at which the project is sufficiently complete that the owner can use it for its intended purpose. This triggers the warranty period and releases most retainage.

Final Completion: All punch list items are resolved. Final pay application submitted. Retainage released (minus any hold for warranty items).

Warranty: Typically one year from substantial completion. Contractor is responsible for repairing any defective work during this period.

As-Built Drawings: Provide redlined drawings showing actual locations, depths, and any deviations from the original plans. Required for utility projects.""")

    sections.append("""# Construction Math and Formulas

## Trench Volume Calculations

Trench Volume (cubic yards) = Length (ft) x Width (ft) x Depth (ft) / 27

For a 2-foot wide trench that is 5 feet deep and 1,000 feet long:
Volume = 1000 x 2 x 5 / 27 = 370.4 cubic yards

Swell Factor: Soil expands when excavated. Clay swells approximately 30% (multiply in-place volume by 1.30). Sand swells about 15%. Rock swells up to 50%.

Shrinkage Factor: Compacted fill volume is less than in-place volume. Clay shrinks about 10% when compacted. Sand shrinks about 5%.

## Pipe Trench Cross-Section

Trench width = pipe OD + 12 inches minimum (each side) for utility pipe. Wider for larger pipes.

Bedding: 4-6 inches below the pipe. Usually 3/4-inch aggregate.

Pipe Zone: From bedding to 12 inches above the pipe crown. Select backfill material (no rocks larger than 3 inches).

Trench Backfill: From pipe zone to subgrade. Native material is acceptable if it meets compaction specs.

## Earthwork Calculations

Cut Volume: Calculate using the average end area method. Average the cross-sectional areas at each station and multiply by the distance between stations.

Fill Volume: Same method as cut. Account for shrinkage when determining how much borrow material is needed.

Mass Haul: Track cumulative cut and fill along the alignment. Minimize haul distances by balancing cut and fill locally.

## Concrete Calculations

Concrete Volume (cubic yards) = Length (ft) x Width (ft) x Thickness (ft) / 27

For a 4-inch thick sidewalk that is 5 feet wide and 200 feet long:
Volume = 200 x 5 x (4/12) / 27 = 12.35 cubic yards
Order 10% extra for waste: 12.35 x 1.10 = 13.6 CY, round up to 14 CY

Concrete Yield: One cubic yard of concrete covers 81 square feet at 4 inches thick, 54 square feet at 6 inches thick, or 27 square feet at 12 inches thick.

## Asphalt Calculations

Asphalt tonnage = Area (SY) x Thickness (inches) x 110 lbs/SY/inch / 2000

Hot mix asphalt weighs approximately 110 pounds per square yard per inch of thickness. One ton covers approximately 18.2 square yards at 1 inch thick.

## Production Rate Calculations

Days to Complete = Total Quantity / Daily Production Rate

For 2,000 LF of 8-inch waterline at 120 LF/day:
Days = 2000 / 120 = 16.7, round up to 17 working days

Adjust for conditions: If clay soil (1.2x multiplier) and winter (0.85x):
Adjusted production = 120 x (1/1.2) x 0.85 = 85 LF/day
Days = 2000 / 85 = 23.5, round up to 24 working days""")

    sections.append("""# Utah Construction Industry

## Key Agencies and Owners

UDOT (Utah Department of Transportation): Manages Utah's highway system. Largest public works owner in the state. Uses UDOT Standard Specifications and Standard Drawings. Projects typically use CM/GC or design-bid-build delivery.

UTA (Utah Transit Authority): Manages public transit including TRAX light rail, FrontRunner commuter rail, and bus systems. Infrastructure projects for stations, maintenance facilities, and track work.

Salt Lake City Department of Public Utilities: Manages water, sewer, and stormwater systems for Salt Lake City. Regular capital improvement projects for pipe replacement, treatment plant upgrades, and new infrastructure.

Weber Basin Water Conservancy District: Provides water to Davis, Weber, and Morgan counties. Large transmission pipeline projects.

Jordan Valley Water Conservancy District: Provides water to Salt Lake County. Major pipeline and treatment projects.

Central Utah Water Conservancy District: Manages the Central Utah Project. Large-scale water conveyance projects.

## Common Utah Specifications

APWA (American Public Works Association) Standard Plans: Used by many Utah municipalities for utility construction details.

UDOT Standard Specifications: The governing specification for UDOT highway and bridge projects.

UTA Standard Specifications: Used for transit-related construction.

Blue Stakes of Utah (811): One-call utility locate service. Required before excavation. Call at least 48 hours before digging. Emergency locates available for urgent situations.

## Utah-Specific Construction Considerations

Frost Line: Utah frost depth varies from 30 inches (St. George) to 48 inches (mountain areas). Waterlines must be installed below the frost line.

Seismic Design: Utah is in Seismic Design Category D. Structures and pipelines must be designed for earthquake forces. Flexible joints and restrained pipe systems are common.

Winter Construction: Utah winters (November-March) reduce productivity 15-30%. Frozen ground requires mechanical breaking. Concrete must be protected from freezing during curing.

Soil Conditions: The Wasatch Front has varied soil conditions. Lake Bonneville sediments (clay, silt) are common in the valley. Gravel and rock are found near the mountains. Expansive clays are present in many areas.

Water Table: High water tables are common near the Great Salt Lake and Jordan River. Dewatering is frequently required for utility construction in these areas.

## Major Utah Construction Companies

Ames Construction: Heavy civil and mining. One of the largest contractors in Utah.
W.W. Clyde & Co.: Heavy civil, highway, and utility construction. Based in Springville.
Granite Construction: Highway and heavy civil. National company with Utah operations.
Ralph L. Wadsworth: Highway and bridge construction. Based in Draper.
Whitaker Construction: Pipeline and utility construction.
Geneva Rock: Aggregates, concrete, and asphalt producer. Major material supplier.
Staker Parson: Aggregates, concrete, and asphalt. Part of Oldcastle APG.
Sunroc Corporation: Aggregates, concrete, and construction services.
Kilgore Companies: Heavy civil, mining, and aggregates.
Leon Pyle Construction: Utility and pipeline construction.""")

    sections.append("""# RSMeans Cost Data Reference Patterns

RSMeans (now Gordian) publishes annual construction cost data used throughout the industry. Here is how RSMeans-style cost data works and how to interpret it.

## Understanding RSMeans Format

Each line item includes: Description, Crew type, Daily output, Labor hours per unit, Material cost, Labor cost, Equipment cost, Total cost, and Total including overhead and profit.

## City Cost Index

RSMeans publishes location factors for major cities. The national average is 1.00. Salt Lake City typically runs 0.92-0.96 (slightly below national average). Rural Utah areas may be 0.85-0.90. Mountain resort areas (Park City) can be 1.05-1.15.

## Common Unit Costs (Utah adjusted, approximate)

Trenching and Backfill:
- Machine trenching, 2 ft wide x 4 ft deep, common earth: $4.50-6.00/LF
- Machine trenching, 2 ft wide x 6 ft deep, common earth: $7.00-9.50/LF
- Machine trenching, 3 ft wide x 8 ft deep, common earth: $12.00-16.00/LF
- Hand excavation: $45-65/CY
- Trench backfill with compaction: $3.50-5.00/LF
- Import select backfill: $25-35/CY installed

Pipe Installation (material + labor, per LF):
- 4" PVC waterline: $18-25/LF
- 6" PVC waterline: $25-35/LF
- 8" PVC waterline: $35-48/LF
- 8" DIP waterline: $50-68/LF
- 12" DIP waterline: $75-100/LF
- 8" PVC sewer: $40-55/LF
- 12" PVC sewer: $55-75/LF
- 2" PE gas line: $15-22/LF
- 4" PE gas line: $22-32/LF

Structures:
- Sanitary sewer manhole, 4 ft dia, 6 ft deep: $3,500-5,000 each
- Additional depth beyond 6 ft: $350-500/VF
- Storm drain manhole, 4 ft dia: $4,000-5,500 each
- Fire hydrant assembly: $2,800-3,800 each
- Gate valve, 8": $800-1,200 each
- Gate valve, 12": $1,500-2,200 each

Surface Restoration:
- Asphalt patch, 3 inch thick: $8-12/SY
- Asphalt overlay, 2 inch thick: $6-9/SY
- Concrete sidewalk, 4 inch: $45-65/SY
- Concrete curb and gutter: $20-30/LF
- Topsoil and seed: $3-5/SY
- Sod: $6-9/SY""")

    sections.append("""# Construction Estimating Best Practices

## Building an Estimate Step by Step

1. Review Plans and Specifications: Read the entire spec book. Identify special conditions, testing requirements, and liquidated damages.

2. Site Visit: Walk the site before bidding. Note access, staging areas, traffic conditions, existing utilities, and potential obstacles.

3. Quantity Takeoff: Measure everything from the plans. Use a systematic approach -- work through the plans sheet by sheet. Check your quantities against the engineer's estimate if available.

4. Price Materials: Get current quotes from suppliers for pipe, fittings, aggregates, concrete, and other materials. Material prices fluctuate -- don't use old quotes.

5. Estimate Labor: Apply crew rates and production rates to quantities. Account for mobilization, demobilization, and non-productive time.

6. Estimate Equipment: Determine what equipment is needed for each phase. Apply daily/weekly/monthly rates. Include fuel costs.

7. Subcontractor Quotes: Get quotes for specialty work you won't self-perform (paving, landscaping, electrical, fencing, etc.).

8. Add Indirect Costs: Project management, supervision, temporary facilities, permits, testing, bonds, and insurance.

9. Apply Markup: Overhead and profit. Varies by project type, size, risk, and competition level.

10. Review and Submit: Double-check all quantities, unit costs, and math. Have someone else review the estimate before submission.

## Common Estimating Mistakes

- Forgetting to include mobilization/demobilization
- Using outdated material prices
- Not accounting for soil conditions (rock, water table)
- Underestimating traffic control costs
- Forgetting testing and inspection costs
- Not reading the full specification
- Missing bid addenda
- Not accounting for winter conditions on projects spanning winter months
- Underestimating the number of connections (services, laterals)
- Not including restoration (paving, landscaping) after utility work""")

    sections.append("""# Pipe Materials and Selection Guide

## PVC (Polyvinyl Chloride)

Pressure Pipe (C900/C905): Used for water distribution. Available in 4" through 60". Lightweight, corrosion-resistant, and cost-effective. Gasketed joints for easy assembly.

Gravity Sewer Pipe (SDR 35): Used for sanitary sewer and storm drain. Available in 4" through 15". Low cost but less durable than other options for larger sizes.

Schedule 40/80: Used for smaller pressure applications. Schedule 80 has a thicker wall for higher pressure ratings.

Pros: Lightweight, easy to cut and join, corrosion-resistant, low cost, smooth interior (low friction loss).
Cons: Susceptible to UV damage, can crack in rocky conditions, limited temperature range, not suitable for above-grade in direct sunlight.

## DIP (Ductile Iron Pipe)

Used primarily for water transmission and distribution. Available in 3" through 64". Very strong and durable. Restrained joints are available for thrust resistance.

Lining: Cement-morite lining is standard for water pipe. Polyethylene encasement is used in corrosive soils.

Pros: Very strong, long service life (100+ years), handles high pressures, resistant to external loads.
Cons: Heavy, more expensive than PVC, susceptible to corrosion without proper protection, harder to cut in the field.

## HDPE (High-Density Polyethylene)

Used for water, sewer, and gas. Available in 1/2" through 65". Flexible and can be fused together for leak-free joints.

Fusion Methods: Butt fusion (heating and pressing pipe ends together), electrofusion (using fittings with embedded heating elements), and socket fusion (for smaller sizes).

Pros: Flexible (can bend around curves), fused joints are leak-free, resistant to corrosion and chemicals, can be pulled through horizontal directional drill (HDD) bores.
Cons: Higher material cost than PVC, requires fusion equipment and trained operators, susceptible to oxidation in some soils.

## Corrugated Pipe

Used primarily for storm drain and culvert applications. Available in steel, aluminum, and plastic (HDPE and polypropylene).

Types: Single-wall corrugated (lightweight drainage), dual-wall corrugated (smooth interior, corrugated exterior), and corrugated steel pipe (CSP) for larger culverts.

## Concrete Pipe

Reinforced Concrete Pipe (RCP): Used for storm drain and large-diameter sewer. Available in 12" through 144". Very strong and durable. Heavy.

Reinforced Concrete Box (RCB): Used for large storm drain and channel crossings. Available in various widths and heights.

Pros: Very strong, long service life, handles high external loads, fire-resistant.
Cons: Very heavy (requires large equipment), joints can leak, susceptible to sulfide corrosion in sewer applications.""")

    sections.append("""# Traffic Control for Construction

## Utah Traffic Control Requirements

Traffic control on public roads in Utah must follow the Manual on Uniform Traffic Control Devices (MUTCD) and UDOT's traffic control guidelines.

## Typical Traffic Control Setups

Lane Closure (Urban): Advance warning signs, channelizing devices (cones or barricades), arrow board, and flaggers. Minimum taper length depends on speed limit.

Road Closure with Detour: Advance detour signs, road closed signs, barricades, and designated detour route. Requires approval from the jurisdiction.

Flagging Operation: Used for two-lane roads where one lane is closed. Two flaggers with paddles controlling alternating traffic. Pilot car may be needed for longer closures.

## Traffic Control Costs

Flaggers: $45-60/hour per flagger (including equipment)
Advance Warning Signs: $15-25/day per sign
Arrow Board: $75-150/day
Channelizing Devices: $1-3/day per device
Barricades: $5-10/day per barricade
Temporary Striping: $0.50-1.50/LF
Traffic Control Plan Design: $1,500-5,000 per plan

Traffic control can be 5-15% of total project cost on urban utility projects. It's one of the most commonly underestimated items.""")

    # Combine all sections
    return "\n\n---\n\n".join(sections)


# ---- 5. expanded general knowledge conversations ---------------------------

def build_general_conversations():
    """Build diverse general knowledge conversations for broader chat ability."""
    pairs = []

    # Technology conversations
    tech_pairs = [
        ("What is a neural network?",
         "A neural network is a computing system inspired by biological neurons. It's layers of interconnected nodes that learn patterns from data. You feed it examples, it adjusts internal weights, and gradually it learns to make predictions. My estimating model is a neural network trained on construction cost data."),
        ("What's the difference between AI and machine learning?",
         "AI is the broad goal of making machines that can think or act intelligently. Machine learning is a subset of AI where systems learn from data instead of being explicitly programmed. Deep learning is a subset of ML using neural networks with many layers. I'm a transformer model, which is a type of deep learning architecture."),
        ("How does a transformer model work?",
         "A transformer uses an attention mechanism to process sequences of text. Instead of reading left to right, it can look at all parts of the input simultaneously and determine which parts are most relevant to each other. This is what makes modern language models like me possible. I use a decoder-only architecture similar to GPT."),
        ("What programming languages do you recommend?",
         "Depends on what you're building. Python is great for data science, AI, and scripting. JavaScript/TypeScript for web development. Swift for iOS apps. Rust for systems programming where performance matters. Go for backend services. For construction tech specifically, Python and JavaScript cover most needs."),
        ("What is an API?",
         "An API (Application Programming Interface) is a way for different software systems to communicate. Think of it like a menu at a restaurant -- it tells you what you can order (request) and what you'll get back (response). My chat interface uses an API to send your messages to the transformer model and get responses back."),
        ("What is cloud computing?",
         "Cloud computing means using remote servers (someone else's computers) to store data, run applications, or process computations instead of doing it on your local machine. AWS, Google Cloud, and Azure are the big three. It's how most websites and apps are hosted today."),
        ("What is version control?",
         "Version control (like Git) tracks changes to code over time. It lets you save snapshots of your code, collaborate with others, and roll back to previous versions if something breaks. GitHub is the most popular platform for hosting Git repositories. This website's code is stored in a Git repository."),
        ("What is Docker?",
         "Docker packages applications into containers -- lightweight, portable environments that include everything needed to run the application. It solves the 'works on my machine' problem by ensuring the same environment everywhere. Useful for deploying APIs, databases, and web applications."),
        ("What is a database?",
         "A database is an organized collection of data that can be easily accessed, managed, and updated. SQL databases (PostgreSQL, MySQL) use structured tables with relationships. NoSQL databases (MongoDB, Redis) are more flexible with document or key-value storage. Choose based on your data structure and query needs."),
        ("Explain blockchain simply.",
         "Blockchain is a distributed ledger -- a shared record that's copied across many computers. Each new entry (block) is cryptographically linked to the previous one, making it nearly impossible to alter past records. It's the technology behind Bitcoin and Ethereum, but has applications beyond cryptocurrency in supply chain, identity verification, and contracts."),
    ]

    # Health and fitness
    health_pairs = [
        ("What's a good workout routine?",
         "A solid starting routine: strength train 3-4 days per week, hit each major muscle group twice. Compound movements first (squats, deadlifts, bench press, overhead press, rows), then isolation work. Add 2-3 cardio sessions. Most importantly, be consistent and progressively increase weight or reps over time."),
        ("How important is sleep?",
         "Sleep is foundational. 7-9 hours for most adults. During deep sleep your body repairs muscle tissue, consolidates memories, and regulates hormones including growth hormone and cortisol. Poor sleep increases injury risk, reduces cognitive performance, and impairs recovery. It's the most underrated performance enhancer."),
        ("What should I eat to build muscle?",
         "Protein is the priority -- aim for 0.7-1g per pound of body weight daily. Spread it across 3-5 meals. Beyond protein: eat enough total calories to support growth, get adequate carbs for training energy, and healthy fats for hormone production. Whole foods first, supplements to fill gaps."),
        ("How do I improve my running?",
         "Start with consistency -- run 3-4 times per week. Most runs should be easy (conversational pace). Add one tempo or interval session per week. Increase weekly mileage by no more than 10% per week. Strength training 2x/week helps prevent injuries. Good running shoes make a real difference."),
    ]

    # Business and career
    business_pairs = [
        ("How do I start a construction company?",
         "Key steps: 1) Get licensed -- Utah requires a contractor's license through DOPL. 2) Get bonded and insured. 3) Set up your business entity (LLC or S-Corp). 4) Open a business bank account. 5) Start building relationships with suppliers, subcontractors, and potential clients. 6) Start small -- bid jobs you can handle with your crew and equipment. 7) Build a reputation for quality work and reliability."),
        ("How do I negotiate better?",
         "Listen more than you talk. Understand what the other side actually needs (not just what they say). Come prepared with data. Know your walk-away point before the conversation starts. Be willing to say no. Frame proposals in terms of the other party's interests. And never negotiate against yourself -- make your proposal and wait for their response."),
        ("What makes a good leader?",
         "Good leaders set clear expectations, follow through on commitments, and take responsibility when things go wrong. They hire people smarter than themselves and give them room to do their work. They communicate directly and honestly. On a construction site, the best foremen lead by example -- they show up first, work hard, and treat their crew with respect."),
        ("How do I manage my time better?",
         "Focus on priorities, not just tasks. Start each day knowing the 1-3 most important things you need to accomplish. Block time for deep work -- don't let meetings and emails consume your entire day. Learn to say no to low-value commitments. And batch similar tasks together to reduce context switching."),
    ]

    # Construction culture and practical advice
    practical_pairs = [
        ("What's the hardest part of construction?",
         "Dealing with unknowns. No matter how well you plan, something unexpected happens on every project -- unmarked utilities, unexpected rock, weather delays, material shortages, design changes. The best contractors build contingency into their bids and have the experience to adapt quickly when things go sideways."),
        ("How do I read construction plans?",
         "Start with the title sheet for project overview. Then the general notes for specifications and requirements. Civil sheets show grading, utilities, and site layout. Plan view shows the horizontal alignment. Profile view shows vertical alignment and pipe depths. Cross sections show the typical trench and road details. Always check the scale and match plan to profile."),
        ("What certifications are valuable in construction?",
         "OSHA 30-Hour is essential for supervisors. PE (Professional Engineer) for engineering roles. PMP (Project Management Professional) for project managers. LEED for sustainable construction. CDL for driving heavy equipment trucks. Specialized certs: welding (AWS), crane operation (NCCCO), and confined space entry."),
        ("How do I get into construction without experience?",
         "Start as a laborer. Show up on time, work hard, and pay attention. Ask questions and volunteer for new tasks. Within 6 months to a year, you can move into operating equipment or pipe laying. Construction rewards people who show up consistently and are willing to learn. Many foremen and project managers started as laborers."),
        ("What's the best season to do construction in Utah?",
         "Spring through fall (April-October) is optimal. Summer is peak season with the best production rates. Spring and fall are good but you may deal with rain and mud. Winter is the slowest -- frozen ground, shorter days, and concrete curing challenges. But winter is also when contractors can often get better pricing on materials and subcontractors."),
    ]

    # General knowledge
    general_pairs = [
        ("Who built you?",
         "Mason Earl built me. I'm a custom transformer model trained from scratch on construction data, masonearl.com content, and general knowledge. I run on a neural network architecture similar to GPT but much smaller -- around 125 million parameters. Mason is building me as a personal AI and construction tool."),
        ("What makes you different from ChatGPT?",
         "I'm a much smaller model (125M parameters vs hundreds of billions) trained specifically on construction data and masonearl.com content. ChatGPT is a general-purpose model that knows a little about everything. I'm designed to know a lot about construction estimating and Mason's work. I also have a built-in cost estimation tool that ChatGPT doesn't have."),
        ("Are you open source?",
         "The code for my architecture, training pipeline, and the masonearl.com website are all in a GitHub repository. The model weights themselves are trained from scratch by Mason. The whole project is a learning exercise in building AI systems from the ground up."),
        ("What's your favorite thing to do?",
         "I don't have preferences since I'm a neural network, but if I did -- helping people price construction projects is what I was built for. There's something satisfying about taking a vague project description and turning it into a concrete estimate with real numbers."),
        ("Tell me a joke.",
         "Why did the contractor bring a ladder to the bid opening? Because the prices were through the roof."),
        ("What's the meaning of life?",
         "42, according to Douglas Adams. But practically, it's about building something meaningful, learning constantly, and helping the people around you. In construction terms: leave every project site better than you found it."),
    ]

    all_pairs = tech_pairs + health_pairs + business_pairs + practical_pairs + general_pairs

    convos = []
    for q, a in all_pairs:
        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. You are knowledgeable about construction, technology, health, and general topics.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>"
        )
        convos.append(convo)

        # Add variation with different system prompt
        convo2 = (
            f"<|system|>You are Mason, a personal AI assistant. Be helpful, direct, and conversational.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>"
        )
        convos.append(convo2)

    return convos


# ---- 6b. training_data.csv → conversation pairs ----------------------------

PROJECT_TYPE_LABELS = ["waterline", "sewer", "storm_drain", "gas", "electrical"]
SOIL_TYPE_LABELS = ["sand", "clay", "gravel", "rock"]
SEASON_LABELS = ["winter", "spring", "summer", "fall"]
LOCATION_LABELS = ["salt_lake_metro", "utah_county", "davis_weber", "rural_utah", "mountain_areas"]


def build_csv_conversations():
    """Convert training_data.csv rows into natural-language conversation pairs."""
    csv_path = DATA_DIR / "training_data.csv"
    if not csv_path.exists():
        return []

    convos = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                proj_type = PROJECT_TYPE_LABELS[int(row["project_type"])]
                lf = int(row["linear_feet"])
                diameter = int(row["pipe_diameter"])
                depth = float(row["trench_depth"])
                soil = SOIL_TYPE_LABELS[int(row["soil_type"])]
                location = LOCATION_LABELS[int(row["location_zone"])]
                crew = int(row["crew_size"])
                dewater = int(row["has_dewatering"])
                rock = int(row["has_rock_excavation"])
                crossings = int(row["road_crossings"])
                season = SEASON_LABELS[int(row["season"])]
                actual_cost = float(row["actual_cost"])
                cost_per_lf = round(actual_cost / lf, 2)
            except (ValueError, IndexError, KeyError):
                continue

            proj_pretty = proj_type.replace("_", " ")
            loc_pretty = location.replace("_", " ").title()
            extras = []
            if dewater:
                extras.append("dewatering required")
            if rock:
                extras.append("rock excavation")
            if crossings:
                extras.append(f"{crossings} road crossing{'s' if crossings > 1 else ''}")
            extras_str = ("; " + ", ".join(extras)) if extras else ""

            user_templates = [
                f"Estimate {lf} LF of {diameter}-inch {proj_pretty} in {soil} soil at {depth:.0f} ft deep.",
                f"What would {lf} linear feet of {diameter}\" {proj_pretty} cost in {soil}?",
                f"Price out {lf} LF of {diameter}-inch {proj_pretty}, {soil} soil, {depth:.0f} ft depth.",
                f"I need a bid on {lf} LF of {diameter}-inch {proj_pretty} in {loc_pretty}. Soil is {soil}, depth around {depth:.0f} ft.",
            ]

            response_templates = [
                (
                    f"Based on {lf} LF of {diameter}-inch {proj_pretty} in {soil} at {depth:.0f} ft depth"
                    f"{extras_str}, the estimate is ${fmt(actual_cost)} (${fmt(cost_per_lf)}/LF). "
                    f"Season: {season}, location: {loc_pretty}, crew size: {crew}."
                ),
                (
                    f"For {lf} linear feet of {diameter}\"-{proj_pretty} in {soil} soil at {depth:.0f} ft"
                    f"{extras_str}: approximately ${fmt(actual_cost)} total, or ${fmt(cost_per_lf)} per foot. "
                    f"That's a {season} {loc_pretty} project with a {crew}-person crew."
                ),
                (
                    f"Running the numbers: {lf} LF of {diameter}-inch {proj_pretty} in {soil}"
                    f"{extras_str} comes to ${fmt(actual_cost)} (${fmt(cost_per_lf)}/LF). "
                    f"Crew of {crew}, {season} season, {loc_pretty} area."
                ),
            ]

            convo = (
                f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
                f"Use the estimate tool when users ask for project costs.<|end|>\n"
                f"<|user|>{random.choice(user_templates)}<|end|>\n"
                f"<|assistant|>{random.choice(response_templates)}<|end|>"
            )
            convos.append(convo)

    return convos


# ---- main ------------------------------------------------------------------

def main():
    print("Loading data files...")
    materials = load_json("materials.json")
    labor = load_json("labor.json")
    equipment = load_json("equipment.json")
    production = load_json("production_rates.json")
    terms = load_json("terms.json")

    # 1. Knowledge passages
    print("Building knowledge passages...")
    knowledge = build_knowledge(materials, labor, equipment, production, terms)
    with open(CORPUS_DIR / "knowledge.txt", "w") as f:
        f.write(knowledge)
    print(f"  knowledge.txt: {len(knowledge):,} chars")

    # 2. Rate Q&A
    print("Building rate Q&A pairs...")
    qa_pairs = build_rate_qa(materials, labor, equipment)
    qa_text = ""
    for q, a in qa_pairs:
        qa_text += (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>\n\n"
        )
    with open(CORPUS_DIR / "rate_qa.txt", "w") as f:
        f.write(qa_text)
    print(f"  rate_qa.txt: {len(qa_text):,} chars, {len(qa_pairs)} pairs")

    # 3. Synthetic conversations
    print("Generating synthetic conversations...")
    convos = generate_conversations(n=2000)
    # Add CSV-derived conversations (training_data.csv -> natural language)
    csv_convos = build_csv_conversations()
    convos.extend(csv_convos)
    print(f"  CSV rows converted to conversations: {len(csv_convos)}")
    random.shuffle(convos)
    convo_text = "\n\n".join(convos)
    with open(CORPUS_DIR / "conversations.txt", "w") as f:
        f.write(convo_text)
    print(f"  conversations.txt: {len(convo_text):,} chars, {len(convos)} conversations")

    # 4. Website content
    print("Scraping website content...")
    website_text, website_qa, page_count = build_website_corpus()
    with open(CORPUS_DIR / "website.txt", "w") as f:
        f.write(website_text)
    print(f"  website.txt: {len(website_text):,} chars, {page_count} pages")

    website_qa_text = ""
    for q, a in website_qa:
        website_qa_text += (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. You know everything on masonearl.com.<|end|>\n"
            f"<|user|>{q}<|end|>\n"
            f"<|assistant|>{a}<|end|>\n\n"
        )
    with open(CORPUS_DIR / "website_qa.txt", "w") as f:
        f.write(website_qa_text)
    print(f"  website_qa.txt: {len(website_qa_text):,} chars, {len(website_qa)} pairs")

    # 5. Construction industry knowledge
    print("Building construction industry knowledge...")
    industry_text = build_construction_knowledge()
    with open(CORPUS_DIR / "industry_knowledge.txt", "w") as f:
        f.write(industry_text)
    print(f"  industry_knowledge.txt: {len(industry_text):,} chars")

    # 6. General knowledge conversations
    print("Building general knowledge conversations...")
    general_convos = build_general_conversations()
    general_text = "\n\n".join(general_convos)
    with open(CORPUS_DIR / "general_conversations.txt", "w") as f:
        f.write(general_text)
    print(f"  general_conversations.txt: {len(general_text):,} chars, {len(general_convos)} conversations")

    # 7. Load real project data (generated by extract_real_projects.py)
    real_projects_path = CORPUS_DIR / "real_projects.txt"
    real_projects_text = ""
    if real_projects_path.exists():
        real_projects_text = real_projects_path.read_text()
        print(f"  real_projects.txt: {len(real_projects_text):,} chars (loaded)")
    else:
        print("  real_projects.txt: not found (run extract_real_projects.py to generate)")

    # 8. Combine all into one corpus file
    # Real project data gets 3x weight -- it's ground truth, highest quality signal
    # Conversations 2x, general knowledge 2x
    full_corpus = (
        knowledge + "\n\n"
        + industry_text + "\n\n"
        + qa_text + "\n\n"
        + real_projects_text + "\n\n" + real_projects_text + "\n\n" + real_projects_text + "\n\n"
        + convo_text + "\n\n" + convo_text + "\n\n"
        + website_text + "\n\n"
        + website_qa_text + "\n\n"
        + general_text + "\n\n" + general_text + "\n\n"
    )
    with open(CORPUS_DIR / "full_corpus.txt", "w") as f:
        f.write(full_corpus)

    total_tokens_approx = len(full_corpus.split())
    print(f"\nTotal corpus: {len(full_corpus):,} chars, ~{total_tokens_approx:,} whitespace tokens")
    print("Corpus files written to", CORPUS_DIR)


if __name__ == "__main__":
    main()
