import json
import random
import argparse
import os
from collections import defaultdict

# ==========================================
# 1. Assets Configuration
# ==========================================

SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William",
    "David", "Richard", "Joseph", "Thomas", "Charles",
    "Christopher", "Daniel", "Matthew", "Anthony", "Mark"
]

FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
    "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Nancy", "Lisa", "Betty", "Margaret", "Sandra"
]

OCCUPATIONS = [
    "Student", "Teacher", "Doctor", "Software Engineer", "Civil Servant",
    "Business Owner", "Factory Worker", "Farmer", "Freelancer", "Retiree"
]

# [New]: Interest pool
INTEREST_POOL = [
    "Sports", "Art", "Technology", "Music", "Reading",
    "Travel", "Gaming", "Cooking"
]


# ==========================================
# 2. Generation Logic
# ==========================================

def generate_english_name(gender):
    """Generate an English-style name."""
    surname = random.choice(SURNAMES)
    given_name = random.choice(MALE_NAMES) if gender == "Male" else random.choice(FEMALE_NAMES)
    return f"{given_name} {surname}"


def generate_demographics():
    """Generate demographic characteristics."""
    gender = random.choice(["Male", "Female"])
    occupation = random.choice(OCCUPATIONS)

    if occupation == "Student":
        age = random.randint(16, 22)
    elif occupation == "Retiree":
        age = random.randint(65, 80)
    else:
        age = random.randint(23, 64)

    return {
        "name": generate_english_name(gender),
        "gender": gender,
        "age": age,
        "occupation": occupation
    }


def generate_truncated_normal(mean=0.5, std_dev=0.15, min_val=0.0, max_val=1.0):
    """Generate a continuous float from a truncated normal distribution."""
    while True:
        val = random.gauss(mean, std_dev)
        if min_val <= val <= max_val:
            return round(val, 3)


def generate_big_five():
    """Generate Big Five personality traits."""
    return {
        "Openness": generate_truncated_normal(),
        "Conscientiousness": generate_truncated_normal(),
        "Extraversion": generate_truncated_normal(),
        "Agreeableness": generate_truncated_normal(),
        "Neuroticism": generate_truncated_normal()
    }


def generate_interests():
    """
    [New]: Randomly assign 1 to 3 interests.
    """
    num_interests = random.randint(1, 3)
    return random.sample(INTEREST_POOL, num_interests)


def generate_single_agent(agent_id):
    """
    Generate single agent data containing all intrinsic attributes.
    """
    demographics = generate_demographics()
    big_five = generate_big_five()
    interests = generate_interests()  # Get interests

    return {
        "id": str(agent_id),
        "demographics": demographics,
        "psychology": {"big_five": big_five},
        "interests": interests  # [New]: Add as an independent module in JSON
    }


# ==========================================
# 3. Validation & Pipeline
# ==========================================

def validate_distribution(agents, scale):
    stats = defaultdict(lambda: defaultdict(int))
    traits_sum = defaultdict(float)
    interest_count = 0

    for agent in agents.values():
        stats["Gender"][agent["demographics"]["gender"]] += 1
        stats["Occupation"][agent["demographics"]["occupation"]] += 1
        interest_count += len(agent["interests"])

        big_five = agent["psychology"]["big_five"]
        for trait, val in big_five.items():
            traits_sum[trait] += val

    print(f"  --> Demographics: {dict(stats['Gender'])}")
    print(f"  --> Avg Interests per Agent: {interest_count / scale:.1f}")


def generate_dataset(num_agents, seed, output_dir):
    random.seed(seed)
    agents = {}
    for i in range(num_agents):
        agents[str(i)] = generate_single_agent(i)

    output_filename = os.path.join(output_dir, f"agents_N{num_agents}_seed{seed}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {num_agents} agents to {output_filename}")
    validate_distribution(agents, num_agents)


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Data Generator for ABM")
    parser.add_argument("--seed", type=int, default=46, help="Random seed")
    parser.add_argument("--out", type=str, default=".", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(f"🚀 Starting dataset generation with Master Seed: {args.seed}")
    print("-" * 50)

    network_scales = [50, 100, 500]
    for scale in network_scales:
        generate_dataset(scale, args.seed, args.out)

    print("-" * 50)
    print("🎉 All datasets generated with interests!")