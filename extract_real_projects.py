#!/usr/bin/env python3
"""Extract real Tempest project data from OneDrive and convert to training conversations.

Reads actual won project bid tabs, pay apps, and pricing data from OneDrive
and generates high-quality natural-language training pairs for the Mason transformer.

Run from the transformer/ directory:
    python extract_real_projects.py
"""

import csv
import json
import os
import random
from pathlib import Path

try:
    import openpyxl
except ImportError:
    print("openpyxl not installed. Run: pip install openpyxl")
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR / "corpus"
CORPUS_DIR.mkdir(exist_ok=True)

ONEDRIVE = Path.home() / "Library" / "CloudStorage" / "Tempest Enterprises - Project Estimating - Documents"
PROJECTS_DIR = ONEDRIVE / "1. All Projects"
RESOURCES_DIR = ONEDRIVE / "2. Resources"

random.seed(42)


def fmt(n):
    try:
        n = float(n)
        if n == int(n):
            return f"{int(n):,}"
        return f"{n:,.2f}"
    except (TypeError, ValueError):
        return str(n)


# ---------------------------------------------------------------------------
# T&M Rates (from 2022 Tempest IHP T&M Rates.pdf)
# These are Tempest's actual T&M billing rates to clients.
# ---------------------------------------------------------------------------

TEMPEST_TM_RATES = {
    "labor": {
        "Project Manager": {"tm_rate": 124.56, "ot_rate": 186.84},
        "Superintendent": {"tm_rate": 79.96, "ot_rate": 119.95},
        "Foreman": {"tm_rate": 67.80, "ot_rate": 101.70},
        "Welder": {"tm_rate": 68.28, "ot_rate": 102.43},
        "Fuser": {"tm_rate": 51.85, "ot_rate": 77.78},
        "Meter Setter": {"tm_rate": 51.80, "ot_rate": 77.70},
        "Backhoe Operator": {"tm_rate": 63.57, "ot_rate": 95.36},
        "Loader Operator": {"tm_rate": 61.85, "ot_rate": 92.78},
        "Directional Bore Machine Operator": {"tm_rate": 68.28, "ot_rate": 102.43},
        "Trencher/Plow Operator": {"tm_rate": 61.85, "ot_rate": 92.78},
        "Sideboom Operator": {"tm_rate": 63.57, "ot_rate": 95.36},
        "Transport Truck Driver": {"tm_rate": 58.87, "ot_rate": 88.30},
        "Dump Truck Driver": {"tm_rate": 49.45, "ot_rate": 74.17},
        "Welder's Helper": {"tm_rate": 51.85, "ot_rate": 77.78},
        "Laborer": {"tm_rate": 47.90, "ot_rate": 71.85},
        "Flag Person": {"tm_rate": 37.00, "ot_rate": 55.50},
        "Mechanic/Parts Chaser": {"tm_rate": 57.69, "ot_rate": 86.53},
    },
    "equipment_hourly": {
        "Cat 325 / JD 270 Excavator (Large)": 93.39,
        "Cat 320 / Cat 225C Excavator (Mid-Large)": 87.55,
        "John Deere 200 Excavator (Mid)": 82.85,
        "Cat 315 / JD 120 Excavator (Small-Mid)": 74.69,
        "Case 580 / JD 310 Rubber Tired Backhoe": 37.34,
        "John Deere 60D Mini Excavator": 37.34,
        "Rock Hammer (attachment)": 60.69,
        "Compactor Plate (attachment)": 30.37,
        "Cat IT28 2.5 cy Rubber Tired Loader": 56.05,
        "Cat 938G 2.5 cy Rubber Tired Loader": 61.88,
        "24\"-36\" Asphalt Zipper": 74.69,
        "Cat D4 Dozer w/Sideboom": 62.45,
        "Cat 561 Pipelayer": 67.72,
        "Cat Challenger Sideboom": 62.45,
        "Air Compressor 185 CFM": 18.70,
        "Air Compressor 375 CFM": 44.37,
        "Rammax Compactor": 25.68,
        "Asphalt Saw w/Truck": 61.88,
        "Asphalt Roller": 31.50,
        "Jumping Jack Compactor": 18.70,
        "Directional Boring Machine JT4020 w/Truck": 246.23,
        "Directional Boring Machine JT920 w/Truck": 155.22,
        "Butt Fusion Machine 10\"-18\"": 62.45,
        "Butt Fusion Machine 6\"-8\"": 36.78,
        "Butt Fusion Machine 4\"": 18.70,
        "Dump Truck": 56.05,
        "Dump Truck with Pup": 74.69,
        "Water Truck 2000 Gallon": 50.21,
        "Welding/Fusing Truck": 44.37,
        "Crew Truck": 23.35,
        "Pickup Truck": 19.55,
    },
}

# ---------------------------------------------------------------------------
# Real Project Bid Data
# ---------------------------------------------------------------------------

REAL_PROJECTS = [
    {
        "name": "Herriman City Old Town Waterline Replacement",
        "owner": "Herriman City",
        "location": "Herriman, Utah",
        "year": 2025,
        "contract_value": 3551512,
        "type": "waterline replacement",
        "scope": "13,196 LF of 8-inch PVC C900 waterline replacement with service laterals, valves, and hydrants",
        "bid_items": [
            {"item": "Mobilization", "unit": "LS", "qty": 1, "unit_price": 80000},
            {"item": "Quality Control & Testing", "unit": "LS", "qty": 1, "unit_price": 35000},
            {"item": "Traffic Control", "unit": "LS", "qty": 1, "unit_price": 19866},
            {"item": "Subsurface Investigation", "unit": "HR", "qty": 80, "unit_price": 130},
            {"item": "Remove & Replace Landscaping", "unit": "LS", "qty": 1, "unit_price": 20476},
            {"item": "Remove & Replace Curb & Gutter", "unit": "LF", "qty": 202, "unit_price": 61.50},
            {"item": "Remove & Replace Sidewalk (6\" Thick)", "unit": "SY", "qty": 130, "unit_price": 195},
            {"item": "Remove Trench Asphalt (4\" Thick)", "unit": "SY", "qty": 3773, "unit_price": 18.50},
            {"item": "Replace Asphalt T-Patch (4\" Thick)", "unit": "TON", "qty": 1053, "unit_price": 284},
            {"item": "Untreated Base Course (8\" Thick)", "unit": "TON", "qty": 1733, "unit_price": 26},
            {"item": "Remove & Replace Gravel Shoulder", "unit": "SY", "qty": 11991, "unit_price": 8.50},
            {"item": "Remove Existing Water Valves", "unit": "EA", "qty": 40, "unit_price": 1416},
            {"item": "Water Mainline Tie-Ins", "unit": "EA", "qty": 30, "unit_price": 6661},
            {"item": "Water Mainline Hot Tap Connection", "unit": "EA", "qty": 1, "unit_price": 14501},
            {"item": "Cut, Cap, & Abandon Existing Waterline", "unit": "EA", "qty": 18, "unit_price": 2202},
            {"item": "Import Pipe Zone Backfill", "unit": "LF", "qty": 13426, "unit_price": 6.30},
            {"item": "Import Pipe Trench Fill", "unit": "LF", "qty": 13426, "unit_price": 9.70},
            {"item": "8\" PVC C900 Pipe", "unit": "LF", "qty": 13196, "unit_price": 46},
            {"item": "10\" PVC C900 Pipe", "unit": "LF", "qty": 230, "unit_price": 184},
            {"item": "6\" Gate Valve (Class 250)", "unit": "EA", "qty": 3, "unit_price": 3261},
            {"item": "8\" Gate Valve (Class 250)", "unit": "EA", "qty": 43, "unit_price": 4163},
            {"item": "10\" Gate Valve (Class 250)", "unit": "EA", "qty": 12, "unit_price": 5748},
            {"item": "16\" Butterfly Valve (Class 250)", "unit": "EA", "qty": 6, "unit_price": 13062},
            {"item": "Remove & Replace Fire Hydrant Assembly", "unit": "EA", "qty": 20, "unit_price": 10844},
            {"item": "Fire Hydrant Laterals", "unit": "LF", "qty": 500, "unit_price": 33},
            {"item": "1\" Service Lateral", "unit": "LF", "qty": 3360, "unit_price": 42},
            {"item": "1\" Service Meter Box & Setter", "unit": "EA", "qty": 103, "unit_price": 2298},
            {"item": "2\" Service Lateral", "unit": "LF", "qty": 160, "unit_price": 52},
            {"item": "2\" Service Meter Box & Setter", "unit": "EA", "qty": 2, "unit_price": 6948},
            {"item": "Install Pressure Reducing Valve (PRV)", "unit": "EA", "qty": 2, "unit_price": 20585},
            {"item": "PRV Electrical", "unit": "LS", "qty": 2, "unit_price": 187035},
            {"item": "Water Line Loop at Utility Crossing", "unit": "EA", "qty": 28, "unit_price": 9232},
        ],
    },
]

# ---------------------------------------------------------------------------
# Historical Bid Data (from Bid Tracking.xlsx - real Tempest bids 2021-2023)
# Type: IHP = intermediate high pressure gas, HP = high pressure gas,
#        SWSD = site water/storm drain, Water = waterline, Misc = miscellaneous
# ---------------------------------------------------------------------------

HISTORICAL_BIDS = [
    # Awarded - IHP gas line work
    {"name": "US-89 Relocation", "location": "Farmington, Utah", "type": "IHP gas relocation", "pipe_size": '6"', "footage": 8325, "services": 16, "bid": 1590894.75, "price_per_lf": 191.10, "awarded": True, "year": 2021, "notes": "1800 LF of 12\" HDD crossing US-89, relocation as part of $250M UDOT highway rebuild"},
    {"name": "Millcreek IHP Replacement", "location": "Salt Lake City, Utah", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 3390, "services": 15, "bid": 741277.90, "price_per_lf": 218.67, "awarded": True, "year": 2021, "notes": "Extremely rocky conditions, lots of difficult services"},
    {"name": "West Temple Belt Line Replacement", "location": "Salt Lake City, Utah", "type": "IHP gas replacement", "pipe_size": '8"', "footage": 8500, "services": 0, "bid": 1066408.45, "price_per_lf": 125.46, "awarded": True, "year": 2021},
    {"name": "American Fork Reinforcement", "location": "American Fork, Utah", "type": "IHP gas reinforcement", "pipe_size": '8"', "footage": 4000, "services": 0, "bid": 252476.88, "price_per_lf": 63.12, "awarded": True, "year": 2021, "notes": "Main reinforcement single sourced"},
    {"name": "WVC IHP Replacement", "location": "Bannock Drive, West Valley, Utah", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 4955, "services": 105, "bid": 679891.55, "price_per_lf": 137.21, "awarded": True, "year": 2021, "notes": "103 service replacements, 2 test and ties, rocky - lost money"},
    {"name": "Harrison Blvd IHP Relocation", "location": "Ogden, Utah", "type": "IHP gas relocation", "pipe_size": '8"', "footage": 1215, "services": 5, "bid": 276994.30, "price_per_lf": 227.98, "awarded": True, "year": 2021, "notes": "Relocation per city project"},
    {"name": "North Ogden IHP Replacement", "location": "400 East, North Ogden, Utah", "type": "IHP gas replacement", "pipe_size": '8"', "footage": 3500, "services": 30, "bid": 992747.50, "price_per_lf": 283.64, "awarded": True, "year": 2021},
    {"name": "Ogden Replacements Bid Package", "location": "Ogden/Clearfield/Uintah, Utah", "type": "IHP gas replacement", "pipe_size": '4"', "footage": 3725, "services": 13, "bid": 436112.75, "price_per_lf": 117.08, "awarded": True, "year": 2021},
    {"name": "Western Replacements Bid Package", "location": "Midvale, Utah", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 6137, "services": 100, "bid": 1091985.40, "price_per_lf": 177.93, "awarded": True, "year": 2021, "notes": "5500 LF of 2\" main and 100+ service replacements in Midvale"},
    {"name": "West Davis UDOT Relocation", "location": "Syracuse, Utah", "type": "IHP gas relocation", "pipe_size": '6"', "footage": 5458, "services": 10, "bid": 1758819, "price_per_lf": 322.25, "awarded": True, "year": 2021, "notes": "UDOT Relocation project"},
    {"name": "South Jordan Reinforcement", "location": "South Jordan, Utah", "type": "IHP gas reinforcement", "pipe_size": '8"', "footage": 4650, "services": 0, "bid": 1077809, "price_per_lf": 231.79, "awarded": True, "year": 2022},
    {"name": "Kilby Road IHP", "location": "Park City, Utah", "type": "IHP gas line", "pipe_size": '4"', "footage": 2400, "services": 9, "bid": 418710, "price_per_lf": 174.46, "awarded": True, "year": 2022},
    {"name": "Monterey Circle IHP Replacement", "location": "Cottonwood Heights, Utah", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 3550, "services": 43, "bid": 770382.50, "price_per_lf": 217.01, "awarded": True, "year": 2022, "notes": "Rocky project, lost money"},
    {"name": "White Mountain Mall Replacement", "location": "Rock Springs, Wyoming", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 13200, "services": 36, "bid": 1163184.25, "price_per_lf": 88.12, "awarded": True, "year": 2021, "notes": "Only 2400 LF underground, rest above ground IPS by third party"},
    {"name": "Dyno Nobel Gas Line", "location": "Tooele, Utah", "type": "IHP gas line", "pipe_size": '4"', "footage": 3240, "services": 3, "bid": 113144, "price_per_lf": 34.92, "awarded": True, "year": 2023},
    {"name": "Shepard Lane UDOT Relocation", "location": "Utah", "type": "IHP gas relocation", "pipe_size": None, "footage": None, "services": None, "bid": 231526, "price_per_lf": None, "awarded": True, "year": 2023},
    # Not awarded - IHP
    {"name": "Midvale IHP Relocation", "location": "Midvale, Utah", "type": "IHP gas relocation", "pipe_size": '6"', "footage": 3015, "services": 35, "bid": 507658.60, "price_per_lf": 168.38, "awarded": False, "year": 2021, "notes": "2% off from winning bid (Dominion)"},
    {"name": "American Fork UDOT Relocation", "location": "American Fork, Utah", "type": "IHP gas relocation", "pipe_size": '2"', "footage": 2725, "services": 33, "bid": 350072.85, "price_per_lf": 128.47, "awarded": False, "year": 2021},
    {"name": "Herriman Highway Reinforcement", "location": "Herriman, Utah", "type": "IHP gas reinforcement", "pipe_size": '6"', "footage": 8130, "services": 31, "bid": 1819570.30, "price_per_lf": 223.81, "awarded": False, "year": 2021},
    {"name": "6200 S UDOT Relocation", "location": "Salt Lake City, Utah", "type": "IHP gas relocation", "pipe_size": '6"', "footage": 2895, "services": 33, "bid": 1224906.75, "price_per_lf": 423.11, "awarded": False, "year": 2022, "notes": "Awarded to Primoris who underbid"},
    {"name": "Salem Arrowhead Reinforcement", "location": "Springville, Utah", "type": "IHP gas reinforcement", "pipe_size": '6"', "footage": 7330, "services": 0, "bid": 783563, "price_per_lf": 106.90, "awarded": False, "year": 2022},
    {"name": "Sheeplane Reinforcement", "location": "Grantsville, Utah", "type": "IHP gas reinforcement", "pipe_size": '8"', "footage": 12600, "services": 0, "bid": 1556779, "price_per_lf": 123.55, "awarded": False, "year": 2022},
    {"name": "Orem IHP Replacement", "location": "Orem, Utah", "type": "IHP gas replacement", "pipe_size": '2"', "footage": 2275, "services": 36, "bid": 613781, "price_per_lf": 269.79, "awarded": False, "year": 2022, "notes": "Fugal awarded, they underbid"},
    {"name": "Springville UDOT Reinforcement", "location": "Springville, Utah", "type": "IHP gas reinforcement", "pipe_size": '8"', "footage": 4025, "services": 0, "bid": 984002.40, "price_per_lf": 244.47, "awarded": False, "year": 2023, "notes": "Lost by under $100K, came in 2nd to Fugal. Included 360 LF of guided auger bore 16\""},
    # Waterline
    {"name": "Kilby Road Irrigation Waterline", "location": "Park City, Utah", "type": "waterline", "pipe_size": '8"', "footage": 3413, "services": 0, "bid": 316466.37, "price_per_lf": 92.72, "awarded": False, "year": 2021, "notes": "Winning bid was $84.68/LF"},
    # Storm drain / site utilities
    {"name": "Morgan Asphalt Lakeside Development", "location": "Jordanelle, Utah", "type": "site utilities (sewer, waterline, storm drain)", "pipe_size": '10"', "footage": 4570, "services": 97, "bid": 669293.15, "price_per_lf": 146.45, "awarded": False, "year": 2021, "notes": "3\" LP sewer, 10\" HDPE waterline, 18\" ADS storm drain; excluded concrete materials"},
    # Misc
    {"name": "Kilgore Private Gas Line", "location": "5400 S Mountain View Corridor, South Jordan, Utah", "type": "private gas line", "pipe_size": '1.25"', "footage": 600, "services": 0, "bid": 10171, "price_per_lf": 16.95, "awarded": True, "year": 2021},
    {"name": "Koch Mechanical Bore", "location": "800 W State St, Farmington, Utah", "type": "HDD bore", "pipe_size": '10" HDPE', "footage": 30, "services": 0, "bid": 4536, "price_per_lf": 151.20, "awarded": True, "year": 2021, "notes": "10\" HDPE bore under the jail"},
]

# ---------------------------------------------------------------------------
# Real aggregate/material pricing from SPC (Staker Parson Companies) 2024
# Tempest pricing (best price tier)
# ---------------------------------------------------------------------------

SPC_AGGREGATE_PRICING = [
    # Beck Street Quarry (Salt Lake City)
    {"material": "2\" minus embankment", "price_per_ton": 5.55, "quarry": "Beck Street, Salt Lake City"},
    {"material": "3/4\" APWA spec road base", "price_per_ton": 20.50, "quarry": "Beck Street, Salt Lake City"},
    {"material": "commercial road base", "price_per_ton": 8.30, "quarry": "Beck Street, Salt Lake City"},
    {"material": "1-1/2\" state spec road base", "price_per_ton": 9.90, "quarry": "Beck Street, Salt Lake City"},
    {"material": "3/8\" natural sand", "price_per_ton": 6.65, "quarry": "Beck Street, Salt Lake City"},
    {"material": "1\" crushed rock", "price_per_ton": 19.55, "quarry": "Beck Street, Salt Lake City"},
    {"material": "1-1/2\" crushed rock", "price_per_ton": 19.55, "quarry": "Beck Street, Salt Lake City"},
    {"material": "3\" minus borrow", "price_per_ton": 9.45, "quarry": "Beck Street, Salt Lake City"},
    {"material": "P-159 (pipe bedding)", "price_per_ton": 8.60, "quarry": "Beck Street, Salt Lake City"},
    {"material": "asphalt millings (RAP)", "price_per_ton": 18.50, "quarry": "Beck Street, Salt Lake City"},
    # Lehi West Quarry
    {"material": "1\" crushed rock", "price_per_ton": 19.40, "quarry": "Lehi West"},
    {"material": "3/4\" crushed rock", "price_per_ton": 19.40, "quarry": "Lehi West"},
    {"material": "commercial road base", "price_per_ton": None, "quarry": "Lehi West"},
    {"material": "1-1/2\" state spec road base", "price_per_ton": 9.55, "quarry": "Lehi West"},
    {"material": "3\" minus borrow", "price_per_ton": 9.25, "quarry": "Lehi West"},
    # Keigley Quarry (Genola)
    {"material": "3/4\" crushed rock", "price_per_ton": 17.80, "quarry": "Keigley, Genola"},
    {"material": "1\" crushed rock", "price_per_ton": 17.80, "quarry": "Keigley, Genola"},
    {"material": "commercial road base", "price_per_ton": 7.60, "quarry": "Keigley, Genola"},
    {"material": "1-1/2\" state spec road base", "price_per_ton": 10.10, "quarry": "Keigley, Genola"},
    {"material": "3/8\" natural sand", "price_per_ton": 5.30, "quarry": "Keigley, Genola"},
    {"material": "asphalt millings (RAP)", "price_per_ton": 16.95, "quarry": "Keigley, Genola"},
    # Grantsville Quarry
    {"material": "commercial road base", "price_per_ton": 7.20, "quarry": "Grantsville"},
    {"material": "1-1/2\" state spec road base", "price_per_ton": 8.55, "quarry": "Grantsville"},
    {"material": "3\" minus borrow", "price_per_ton": 7.35, "quarry": "Grantsville"},
    {"material": "1\" crushed rock", "price_per_ton": 16.35, "quarry": "Grantsville"},
]

# ---------------------------------------------------------------------------
# Real production rates from WW Clyde Estimating Calculator
# LF/hr by excavator type for gas/utility line work
# ---------------------------------------------------------------------------

PRODUCTION_RATES = [
    {"excavator": "Case 580 / JD 310 Rubber Tired Backhoe", "excavating_lf_hr": 6, "backfilling_lf_hr": 43, "pipe_laying_lf_hr": 34, "net_production_lf_hr": 6, "daily_8hr": 33, "daily_10hr": 45},
    {"excavator": "Cat 308 Mini Excavator", "excavating_lf_hr": 10, "backfilling_lf_hr": 76, "pipe_laying_lf_hr": 63, "net_production_lf_hr": 15, "daily_8hr": 82.5, "daily_10hr": 112.5},
    {"excavator": "Cat 315 Excavator", "excavating_lf_hr": 21, "backfilling_lf_hr": 156, "pipe_laying_lf_hr": 134, "net_production_lf_hr": 21, "daily_8hr": 115.5, "daily_10hr": 157.5},
    {"excavator": "Cat 320 Excavator", "excavating_lf_hr": 22, "backfilling_lf_hr": 156, "pipe_laying_lf_hr": 137, "net_production_lf_hr": 22, "daily_8hr": 121, "daily_10hr": 165},
    {"excavator": "Cat 330 Excavator", "excavating_lf_hr": 31, "backfilling_lf_hr": 219, "pipe_laying_lf_hr": 193, "net_production_lf_hr": 31, "daily_8hr": 170.5, "daily_10hr": 232.5},
]

# Southwest Gas unit prices from Storm AI contractor proposal
SWG_UNIT_PRICES = [
    {"item": '2" PE directional bore under paved surface', "unit": "FT", "unit_price": 15.00, "source": "Southwest Gas contract"},
    {"item": '2" PE open trench under asphalt', "unit": "FT", "unit_price": 8.00, "source": "Southwest Gas contract"},
    {"item": '2" PE open trench under concrete', "unit": "FT", "unit_price": 12.00, "source": "Southwest Gas contract"},
    {"item": '1" PE directional bore improved landscape', "unit": "FT", "unit_price": 12.00, "source": "Southwest Gas contract"},
    {"item": "Slurry seal / MicroSeal application", "unit": "SQ.YD", "unit_price": 1.25, "source": "Southwest Gas contract"},
    {"item": "Three man crew with backhoe (T&M)", "unit": "DAY", "unit_price": 3500, "source": "Southwest Gas T&M"},
    {"item": "Service abandonment", "unit": "EA", "unit_price": None, "source": "Southwest Gas contract"},
    {"item": "As-built / FOMS data entry", "unit": "EA", "unit_price": 36000, "source": "Southwest Gas contract"},
    {"item": "Mill and overlay rubberized asphalt", "unit": "SQ.YD", "unit_price": 8.00, "source": "Southwest Gas estimate"},
    {"item": "Traffic control (voucher)", "unit": "MOB", "unit_price": 48000, "source": "Southwest Gas contract"},
]


# ---------------------------------------------------------------------------
# Conversation generators
# ---------------------------------------------------------------------------

def build_tm_rate_conversations():
    """Generate Q&A about Tempest's actual T&M billing rates."""
    convos = []

    for role, rates in TEMPEST_TM_RATES["labor"].items():
        tm = rates["tm_rate"]
        ot = rates["ot_rate"]

        q_templates = [
            f"What is the T&M rate for a {role.lower()}?",
            f"How much does Tempest charge for a {role.lower()} on T&M work?",
            f"What's the billing rate for a {role.lower()}?",
            f"What do we charge per hour for a {role.lower()}?",
        ]
        a_templates = [
            f"The T&M billing rate for a {role.lower()} is ${fmt(tm)}/hr regular time, ${fmt(ot)}/hr overtime.",
            f"On T&M work, a {role.lower()} bills at ${fmt(tm)} per hour (${fmt(ot)}/hr OT).",
        ]

        for _ in range(2):
            convo = (
                f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
                f"You have access to Tempest Enterprises' actual labor and equipment rates.<|end|>\n"
                f"<|user|>{random.choice(q_templates)}<|end|>\n"
                f"<|assistant|>{random.choice(a_templates)}<|end|>"
            )
            convos.append(convo)

    for equip, rate in TEMPEST_TM_RATES["equipment_hourly"].items():
        q_templates = [
            f"What's the hourly rate for a {equip.lower()}?",
            f"How much does a {equip.lower()} cost per hour on T&M?",
            f"What do we charge for a {equip.lower()}?",
        ]
        a_templates = [
            f"The {equip} bills at ${fmt(rate)}/hr on T&M work.",
            f"${fmt(rate)}/hr is the T&M rate for a {equip.lower()}.",
        ]
        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
            f"You have access to Tempest Enterprises' actual equipment billing rates.<|end|>\n"
            f"<|user|>{random.choice(q_templates)}<|end|>\n"
            f"<|assistant|>{random.choice(a_templates)}<|end|>"
        )
        convos.append(convo)

    return convos


def build_project_bid_conversations():
    """Generate Q&A from real won project bid tabs."""
    convos = []

    for proj in REAL_PROJECTS:
        name = proj["name"]
        owner = proj["owner"]
        loc = proj["location"]
        value = proj["contract_value"]
        scope = proj["scope"]
        items = proj["bid_items"]

        # Project overview conversations
        overview_questions = [
            f"Tell me about the {name} project.",
            f"What was the scope of the {owner} project?",
            f"How much was the {owner} contract worth?",
            f"Describe the {name} job.",
        ]
        overview_answer = (
            f"The {name} was a {proj['type']} project awarded to Tempest Enterprises by {owner} in {proj['year']}. "
            f"Contract value: ${fmt(value)}. Scope: {scope}. Location: {loc}."
        )
        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
            f"You have detailed knowledge of Tempest Enterprises' won projects and bid history.<|end|>\n"
            f"<|user|>{random.choice(overview_questions)}<|end|>\n"
            f"<|assistant|>{overview_answer}<|end|>"
        )
        convos.append(convo)

        # Line item unit price conversations
        for item_data in items:
            item = item_data["item"]
            unit = item_data["unit"]
            qty = item_data["qty"]
            up = item_data["unit_price"]
            total = qty * up

            if unit in ("LS",) or up is None:
                # Lump sum items - just reference the total
                q_templates = [
                    f"How much did Tempest bid for {item.lower()} on the {owner} project?",
                    f"What was the {item.lower()} line item on the {owner} job?",
                ]
                a_templates = [
                    f"On the {name} project, {item} was bid at ${fmt(total)} lump sum.",
                    f"The {item} line item on the {owner} job was ${fmt(total)} LS.",
                ]
            else:
                q_templates = [
                    f"What unit price did Tempest use for {item.lower()} on the {owner} job?",
                    f"How much per {unit} did we bid for {item.lower()}?",
                    f"What's the unit cost for {item.lower()} on the {owner} project?",
                ]
                a_templates = [
                    f"On the {name} project, {item} was bid at ${fmt(up)}/{unit} ({qty:,} {unit} x ${fmt(up)} = ${fmt(total)}).",
                    f"Tempest used ${fmt(up)}/{unit} for {item.lower()} on the {owner} job. Total: ${fmt(total)} for {qty:,} {unit}.",
                ]

            convo = (
                f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
                f"You have access to Tempest Enterprises' project bid history and unit prices.<|end|>\n"
                f"<|user|>{random.choice(q_templates)}<|end|>\n"
                f"<|assistant|>{random.choice(a_templates)}<|end|>"
            )
            convos.append(convo)

        # Comparison/lookup conversations about major items
        pipe_items = [i for i in items if "pipe" in i["item"].lower() or "PVC" in i["item"] or "DIP" in i["item"]]
        if pipe_items:
            for pipe in pipe_items:
                convo = (
                    f"<|system|>You are Mason, a personal AI assistant. You know Tempest's actual bid prices.<|end|>\n"
                    f"<|user|>What did {pipe['item']} cost per LF on a recent waterline job?<|end|>\n"
                    f"<|assistant|>On the {name} ({owner}, {proj['year']}), {pipe['item']} was ${fmt(pipe['unit_price'])}/LF. "
                    f"That's a real Tempest bid unit price for a city waterline replacement project.<|end|>"
                )
                convos.append(convo)

    return convos


def build_swg_conversations():
    """Generate conversations from Southwest Gas contract pricing."""
    convos = []

    for item in SWG_UNIT_PRICES:
        if item["unit_price"] is None:
            continue
        desc = item["item"]
        unit = item["unit"]
        price = item["unit_price"]
        source = item["source"]

        q_templates = [
            f"What does {desc.lower()} cost?",
            f"What's the unit price for {desc.lower()}?",
            f"How much is {desc.lower()} on a Southwest Gas job?",
        ]
        a_templates = [
            f"Based on {source}, {desc} runs ${fmt(price)}/{unit}.",
            f"${fmt(price)}/{unit} for {desc.lower()} per {source}.",
        ]

        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
            f"You have Tempest's Southwest Gas contract pricing data.<|end|>\n"
            f"<|user|>{random.choice(q_templates)}<|end|>\n"
            f"<|assistant|>{random.choice(a_templates)}<|end|>"
        )
        convos.append(convo)

    return convos


def build_knowledge_passages():
    """Build prose knowledge passages from real project and rate data."""
    sections = []

    # T&M rates passage
    labor_lines = []
    for role, rates in TEMPEST_TM_RATES["labor"].items():
        labor_lines.append(
            f"{role}: ${fmt(rates['tm_rate'])}/hr regular, ${fmt(rates['ot_rate'])}/hr OT"
        )
    equip_lines = []
    for equip, rate in TEMPEST_TM_RATES["equipment_hourly"].items():
        equip_lines.append(f"{equip}: ${fmt(rate)}/hr")

    sections.append(
        "# Tempest Enterprises T&M Rate Schedule (2022)\n\n"
        "These are Tempest Enterprises' actual time-and-materials billing rates to clients. "
        "All material is invoiced at cost plus 10%. Rental equipment without fuel/maintenance invoiced at cost plus 10%.\n\n"
        "## Labor Rates (Hourly)\n"
        + "\n".join(labor_lines)
        + "\n\n## Equipment Rates (Hourly)\n"
        + "\n".join(equip_lines)
    )

    # Real project data passage
    for proj in REAL_PROJECTS:
        item_lines = []
        for item in proj["bid_items"]:
            total = item["qty"] * item["unit_price"]
            item_lines.append(
                f"- {item['item']}: ${fmt(item['unit_price'])}/{item['unit']} "
                f"(qty: {item['qty']:,}, total: ${fmt(total)})"
            )
        sections.append(
            f"# {proj['name']} - Bid Breakdown\n\n"
            f"Owner: {proj['owner']} | Location: {proj['location']} | Year: {proj['year']}\n"
            f"Contract Value: ${fmt(proj['contract_value'])}\n"
            f"Scope: {proj['scope']}\n\n"
            "## Bid Line Items\n"
            + "\n".join(item_lines)
        )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Historical bid conversation generator
# ---------------------------------------------------------------------------

def build_historical_bid_conversations():
    """Generate Q&A from historical Tempest bid data."""
    convos = []

    for bid in HISTORICAL_BIDS:
        name = bid["name"]
        loc = bid["location"]
        btype = bid["type"]
        awarded = bid["awarded"]
        amount = bid["bid"]
        ppf = bid.get("price_per_lf")
        footage = bid.get("footage")
        services = bid.get("services")
        pipe = bid.get("pipe_size")
        notes = bid.get("notes", "")
        year = bid.get("year", "")
        status = "awarded to Tempest" if awarded else "not awarded (bid submitted)"

        # Overview conversation
        overview_q = [
            f"Tell me about the {name} project.",
            f"What was the {name} bid?",
            f"Did Tempest win the {name} job?",
        ]
        overview_parts = [f"The {name} project ({loc}, {year}) was a {btype} job."]
        if footage and ppf:
            overview_parts.append(f"Tempest bid ${fmt(amount)} total (${fmt(ppf)}/LF) for {footage:,} LF" + (f" of {pipe}" if pipe else "") + f". {status.capitalize()}.")
        elif footage:
            overview_parts.append(f"Footage: {footage:,} LF. Total bid: ${fmt(amount)}. {status.capitalize()}.")
        else:
            overview_parts.append(f"Total bid: ${fmt(amount)}. {status.capitalize()}.")
        if notes:
            overview_parts.append(notes)
        overview_a = " ".join(overview_parts)

        convo = (
            f"<|system|>You are Mason, a personal AI assistant built by Mason Earl. "
            f"You have detailed knowledge of Tempest Enterprises' historical bid data.<|end|>\n"
            f"<|user|>{random.choice(overview_q)}<|end|>\n"
            f"<|assistant|>{overview_a}<|end|>"
        )
        convos.append(convo)

        # Unit price conversation
        if ppf and footage and pipe:
            price_q = [
                f"What does {pipe} {btype.replace('IHP ', '').replace('gas ', '')} cost per foot in {loc.split(',')[1].strip() if ',' in loc else loc}?",
                f"What's a realistic price per LF for {pipe} {btype}?",
                f"How much did Tempest bid per foot for {pipe} gas line work?",
            ]
            price_a = [
                f"On the {name} project ({loc}, {year}), Tempest bid ${fmt(ppf)}/LF for {footage:,} LF of {pipe} {btype}. Total: ${fmt(amount)}. {'Awarded.' if awarded else 'Not awarded.'}",
                f"${fmt(ppf)}/LF for {pipe} {btype} at {name} ({year}). {footage:,} LF, total ${fmt(amount)}. {'Won.' if awarded else 'Did not win.'}",
            ]
            convo2 = (
                f"<|system|>You are Mason, a personal AI assistant. You know Tempest's actual historical bid prices.<|end|>\n"
                f"<|user|>{random.choice(price_q)}<|end|>\n"
                f"<|assistant|>{random.choice(price_a)}<|end|>"
            )
            convos.append(convo2)

    # Summary conversations (price ranges by type)
    ihp_awarded = [b for b in HISTORICAL_BIDS if b["awarded"] and b["type"].startswith("IHP") and b.get("price_per_lf") and b.get("pipe_size")]
    if ihp_awarded:
        min_ppf = min(b["price_per_lf"] for b in ihp_awarded)
        max_ppf = max(b["price_per_lf"] for b in ihp_awarded)
        avg_ppf = sum(b["price_per_lf"] for b in ihp_awarded) / len(ihp_awarded)
        convo_summary = (
            f"<|system|>You are Mason, a personal AI assistant. You know Tempest's actual bid history.<|end|>\n"
            f"<|user|>What's a typical price range for IHP gas line work per foot?<|end|>\n"
            f"<|assistant|>Based on Tempest's historical bids, IHP gas line work typically runs ${fmt(min_ppf)}-${fmt(max_ppf)}/LF, with an average around ${fmt(avg_ppf)}/LF. "
            f"Factors like pipe size, soil conditions, urban vs. rural, and UDOT involvement significantly affect price. "
            f"Rocky soil and UDOT relocations push toward the higher end. Simple open-field reinforcements can be as low as ${fmt(min_ppf)}/LF.<|end|>"
        )
        convos.append(convo_summary)

    return convos


def build_aggregate_conversations():
    """Generate Q&A from SPC 2024 aggregate pricing."""
    convos = []

    for item in SPC_AGGREGATE_PRICING:
        if item["price_per_ton"] is None:
            continue
        mat = item["material"]
        price = item["price_per_ton"]
        quarry = item["quarry"]

        q_templates = [
            f"How much does {mat} cost?",
            f"What's the price per ton for {mat} from SPC?",
            f"What does Tempest pay for {mat}?",
        ]
        a_templates = [
            f"Per SPC's 2024 Tempest pricing, {mat} is ${fmt(price)}/ton from the {quarry} quarry.",
            f"${fmt(price)}/ton for {mat} at Tempest rates from SPC's {quarry} location (2024).",
        ]

        convo = (
            f"<|system|>You are Mason, a personal AI assistant. You have Tempest's 2024 SPC aggregate pricing.<|end|>\n"
            f"<|user|>{random.choice(q_templates)}<|end|>\n"
            f"<|assistant|>{random.choice(a_templates)}<|end|>"
        )
        convos.append(convo)

    return convos


def build_production_rate_conversations():
    """Generate Q&A from WW Clyde production rate calculator data."""
    convos = []

    for rate in PRODUCTION_RATES:
        equip = rate["excavator"]
        daily_8 = rate["daily_8hr"]
        daily_10 = rate["daily_10hr"]
        net = rate["net_production_lf_hr"]

        q_templates = [
            f"How many linear feet per day can a {equip} install?",
            f"What production rate should I use for a {equip}?",
            f"What's the LF/day production for a {equip} on utility work?",
        ]
        a_templates = [
            f"A {equip} produces approximately {daily_8} LF/day on an 8-hour day, or {daily_10} LF/day on a 10-hour day for standard utility pipe work. Net rate: {net} LF/hr (limited by excavating). This assumes hard soil conditions, roadway work with shoring, and utility congestion.",
            f"For {equip}: {daily_8} LF/8-hr day or {daily_10} LF/10-hr day. Limiting factor is excavating at {net} LF/hr. Conditions: hard soil, roadway, shoring required, utility conflicts.",
        ]

        convo = (
            f"<|system|>You are Mason, a personal AI assistant. You have real production rate data from Tempest's estimating calculator.<|end|>\n"
            f"<|user|>{random.choice(q_templates)}<|end|>\n"
            f"<|assistant|>{random.choice(a_templates)}<|end|>"
        )
        convos.append(convo)

    # General production rate comparison conversation
    convo_compare = (
        f"<|system|>You are Mason, a personal AI assistant. You know Tempest's actual production rates.<|end|>\n"
        f"<|user|>How do different excavators compare for production on gas line work?<|end|>\n"
        f"<|assistant|>Based on Tempest's production rate calculator for standard utility pipe work (hard soil, roadway, shoring required):\n"
        f"- Case 580 backhoe: 33 LF/8hr day (limited by 6 LF/hr excavating)\n"
        f"- Cat 308 mini: 82 LF/8hr day (15 LF/hr net)\n"
        f"- Cat 315: 115 LF/8hr day (21 LF/hr net)\n"
        f"- Cat 320: 121 LF/8hr day (22 LF/hr net)\n"
        f"- Cat 330: 170 LF/8hr day (31 LF/hr net)\n"
        f"The Case 580 is much slower -- use it only for small service work. Cat 315/320 are the standard for 2-8 inch utility mains.<|end|>"
    )
    convos.append(convo_compare)

    return convos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building real project training data...")

    # 1. T&M rate conversations
    tm_convos = build_tm_rate_conversations()
    print(f"  T&M rate conversations: {len(tm_convos)}")

    # 2. Real project bid conversations (Herriman bid tab)
    bid_convos = build_project_bid_conversations()
    print(f"  Project bid tab conversations: {len(bid_convos)}")

    # 3. Southwest Gas contract pricing conversations
    swg_convos = build_swg_conversations()
    print(f"  Southwest Gas pricing conversations: {len(swg_convos)}")

    # 4. Historical bid data (60+ real Tempest bids 2021-2023)
    hist_convos = build_historical_bid_conversations()
    print(f"  Historical bid conversations: {len(hist_convos)}")

    # 5. SPC aggregate pricing
    agg_convos = build_aggregate_conversations()
    print(f"  Aggregate pricing conversations: {len(agg_convos)}")

    # 6. Production rate conversations
    prod_convos = build_production_rate_conversations()
    print(f"  Production rate conversations: {len(prod_convos)}")

    # 7. Knowledge passages (prose)
    knowledge = build_knowledge_passages()
    print(f"  Knowledge passages: {len(knowledge):,} chars")

    all_convos = tm_convos + bid_convos + swg_convos + hist_convos + agg_convos + prod_convos
    random.shuffle(all_convos)
    convo_text = "\n\n".join(all_convos)

    # Write corpus - 2x weight on conversations (real project data, highest quality signal)
    output = knowledge + "\n\n" + convo_text + "\n\n" + convo_text
    out_path = CORPUS_DIR / "real_projects.txt"
    out_path.write_text(output)
    print(f"\nWrote {len(output):,} chars -> {out_path}")
    print(f"Total conversations: {len(all_convos)}")


if __name__ == "__main__":
    main()
