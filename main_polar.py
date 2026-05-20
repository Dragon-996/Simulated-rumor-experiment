import json
import random
import time
import os
import logging
import math
from datetime import datetime
from prompt_polar import generate_agent_prompt
from LLM_call import LLMInterface

# Top of main_run.py

DAILY_CORPUS = {
    "Student": ["So tired of the 8 AM class, want to sleep.", "Too much homework, going bald.", "The new dish in the cafeteria is pretty good.", "Finally class is over, going back to the dorm to play games!"],
    "Teacher": ["Just finished grading homework, ready to get off work.", "Have to prepare for a public lecture next week, mountain of pressure.", "Three classes in a row, my throat is smoking."],
    "Farmer": ["Looking at the weather it should rain tomorrow, good thing I don't need to water the fields.", "The harvest this year should be okay.", "Working on the farm all day, back aches."],
    "Civil Servant": ["Another day of writing materials, headache.", "Just finished a meeting, a bunch of directives need to be implemented.", "Off work on time, life is good."],
    "Factory Worker": ["The assembly line is moving so fast today.", "Finally made it to the shift change, going to the cafeteria.", "Got paid, treating myself tonight!"],
    "Doctor": ["Saw so many patients in the clinic today, no time to drink water.", "Doctor-patient communication is truly an art.", "Night shift, pray for peace."],
    "Business Owner": ["Business is hard, but still have to persist.", "It's the end of the month, looking at the financial statements makes me worry.", "Closed a small deal today."],
    "Freelancer": ["The client asked to change the requirements again, it's already the fifth version!", "Freelancing means being your own boss.", "Finally submitted the draft, lying flat for two days."],
    "Retiree": ["Did some Tai Chi in the park, feeling refreshed.", "Young people nowadays work too hard.", "Going to pick up my grandson from school in the afternoon."],
    "Software Engineer": ["Code runs! Zero bugs!", "Another day of fixing legacy code.", "Planning not to work overtime today, time to run!"]
}
# ==========================================
# 1. Logging Configuration
# ==========================================
def setup_logging(results_dir):
    logger = logging.getLogger("SocialSimulation")
    logger.setLevel(logging.DEBUG)

    # ==========================================
    # [New]: Clear old log handlers to prevent duplicate logging during Monte Carlo loops
    # ==========================================
    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(results_dir, "simulation.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    # [Modified]: Change INFO to DEBUG to ensure the console outputs detailed node tracking info
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ==========================================
# 2. Simulation Core Engine
# ==========================================
class SocialSimulation:
    def __init__(self, config):
        self.config = config
        self.current_round = 0
        self.simulation_start_time = datetime.now()
        self.timestamp = self.simulation_start_time.strftime("%Y%m%d-%H%M%S")
        self.results_dir = f"simulation_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.logger = setup_logging(self.results_dir)
        self.logger.info(f"=== Complex Network Multi-Agent Social Simulation System Initialization ===")

        # ==========================================
        # [New]: Set global random seed to ensure complete reproducibility
        # ==========================================
        self.seed = config.get('seed', 46)  # Default to 46 if not provided
        random.seed(self.seed)
        self.logger.info(f"Global random seed set to: {self.seed}")

        # Load data and interfaces
        self.agents = self.load_agents(config['agent_file'])
        self.total_agents = len(self.agents)
        self.llm_interface = LLMInterface(
            model_provider=config['model_provider'],
            api_key=config['api_key'],
            model_name=config['model_name'],
            temperature=config.get('temperature', 0.7)
        )

        # State containers
        self.current_posts = {}
        self.agent_beliefs = {aid: "neutral" for aid in self.agents}
        self.post_history = []

        self.simulation_results = {
            "config": config,
            "start_time": self.simulation_start_time.isoformat(),
            "rounds": []
        }

        self.initialize_posts()

    def load_agents(self, file_path):
        self.logger.info(f"Loading network agent data: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['agents']

    def initialize_posts(self):
        """Initialize world and seed node injection"""
        self.logger.info("Initializing timeline and seed nodes...")
        neutral_posts = ["The weather is really nice today.", "Work is so tiring, want a vacation.", "Planning to watch a movie this weekend."]
        self.initial_rumor = self.config.get('initial_rumor', "Urgent notice! Internal news says there will be massive layoffs soon, run!")

        # 1. Round 0: Neutral posts from everyone
        for agent_id in self.agents:
            initial_post = random.choice(neutral_posts)
            self.current_posts[agent_id] = initial_post
            self.post_history.append((0, agent_id, initial_post))

        # 2. Strategically select seed nodes
        all_agents = list(self.agents.keys())
        num_seeds = max(1, int(self.total_agents * self.config.get('initial_spreader_ratio', 0.05)))
        strategy = self.config.get('seed_strategy', 'random')

        if strategy == 'structural':
            self.logger.info("Using [Topological Centrality Influencer] strategy to select seeds...")
            sorted_by_degree = sorted(all_agents, key=lambda aid: len(self.agents[aid].get("neighbors", [])),
                                      reverse=True)
            candidate_spreaders = sorted_by_degree[:num_seeds]
        elif strategy == 'psychological':
            self.logger.info("Using [Psychologically Susceptible] strategy to select seeds...")

            def psy_score(aid):
                psy = self.agents[aid]["psychology"]["big_five"]
                return psy["Extraversion"] + psy["Neuroticism"]

            sorted_by_psy = sorted(all_agents, key=psy_score, reverse=True)
            candidate_spreaders = sorted_by_psy[:num_seeds]
        else:
            self.logger.info("Using [Random Sampling] strategy to select seeds...")
            candidate_spreaders = random.sample(all_agents, num_seeds)

        # 3. Infect seeds (Split in half, two-faction confrontation)
        half_idx = len(candidate_spreaders) // 2
        support_seeds = candidate_spreaders[:half_idx]
        oppose_seeds = candidate_spreaders[half_idx:]

        # Read pros and cons copy from config (use default if not found)
        post_support = self.config.get('initial_support', "[Opinion] Strongly support AI replacing tedious work, humans should be liberated!")
        post_oppose = self.config.get('initial_oppose',
                                          "[Rebuttal] Strongly oppose! Large-scale replacement of jobs by AI will lead to severe unemployment and social crisis!")

        # 3.1 Inject support seeds
        for agent_id in support_seeds:
            self.current_posts[agent_id] = post_support
            self.agent_beliefs[agent_id] = "support"
            self.post_history = [(r, aid, p) if aid != agent_id else (0, aid, post_support) for r, aid, p in
                                    self.post_history]

        # 3.2 Inject oppose seeds
        for agent_id in oppose_seeds:
            self.current_posts[agent_id] = post_oppose
            self.agent_beliefs[agent_id] = "oppose"
            self.post_history = [(r, aid, p) if aid != agent_id else (0, aid, post_oppose) for r, aid, p in
                                    self.post_history]

    def run_simulation(self, rounds=3):
        self.logger.info(f"\n========== Starting Multi-Agent Simulation Evolution ==========")
        for round_idx in range(rounds):
            self.current_round = round_idx + 1
            start_time = time.time()
            self.logger.info(f"\n► Executing round: {self.current_round} / {rounds}")

            round_result = self.run_round()
            self.simulation_results["rounds"].append(round_result)
            self.save_round_snapshot(round_result)

            duration = time.time() - start_time
            stats = round_result["summary"]

            # ==========================================
            # [Fixed here]: Align key names with the latest 4-quadrant statistical metrics
            # ==========================================
            self.logger.info(
                f"[Round {self.current_round} Summary] Time: {duration:.1f}s | "
                f"Topic Participation: {stats['penetration_rate']:.1%} | "
                f"Support Ratio: {stats['support_ratio']:.1%} | "
                f"Oppose Ratio: {stats['oppose_ratio']:.1%} | "
                f"Neutral Ratio: {stats['neutral_ratio']:.1%} | "
                f"Sys Entropy: {stats['system_entropy']:.3f} | "
                f"Bipolarization: {stats['bipolarization_index']:.3f}"
            )

        self.save_final_results()
        self.logger.info("\n========== Social Simulation Successfully Completed ==========")
        return self.simulation_results

    def run_round(self):
        """Synchronous state update with memory (Polarization exclusive version)"""
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)

        round_data = {
            "round": self.current_round,
            "agent_decisions": {},
            "summary": {}
        }

        # Generate static world snapshot
        snapshot_posts = self.current_posts.copy()
        next_round_posts = {}

        # ==========================================
        # Polarization three-faction counter
        # ==========================================
        counts = {"support": 0, "oppose": 0, "neutral": 0}

        for idx, agent_id in enumerate(agent_ids):
            agent = self.agents[agent_id]
            agent_history = [(r, p) for r, aid, p in self.post_history if aid == agent_id]

            self.logger.debug(f"\n{'-' * 80}")
            self.logger.info(
                f"▶ [Round {self.current_round}] Processing node {idx + 1}/{self.total_agents} (ID: {agent_id} - {agent['demographics']['name']})")

            try:
                # 1. Get model response (No longer has debunking and physical isolation interception)
                prompt_data = generate_agent_prompt(agent_id, self.agents, snapshot_posts, agent_history)
                final_prompt = prompt_data["prompt"]

                self.logger.debug(f"  [State 1] Own post history: {prompt_data.get('self_history', [])}")
                self.logger.debug(f"  [State 2] Seen neighbor posts: {prompt_data.get('seen_posts', [])}")
                self.logger.debug(f"  [State 3] Complete Sent Prompt:\n{final_prompt}")

                response = self.llm_interface.call_llm(final_prompt)

                self.logger.debug(f"  [State 4] Final parsed behavior dictionary: {json.dumps(response, ensure_ascii=False)}")

                # ==========================================
                # [Polarization exclusive logic]: Extract stance and fault tolerance
                # ==========================================
                stance = str(response.get("stance", "neutral")).lower()
                if stance not in ["support", "oppose", "neutral"]:
                    stance = "neutral"  # Fault tolerance: If LLM fills in randomly, default to neutral

                will_spread = response.get("will_spread", False)
                new_post = str(response.get("new_post", "")).strip()

                # --- State Record ---
                self.agent_beliefs[agent_id] = stance

                # --- Behavior Record ---
                if will_spread and new_post:
                    next_round_posts[agent_id] = new_post
                    self.post_history.append((self.current_round, agent_id, new_post))
                    self.logger.info(f"  ★ [{agent['demographics']['name']}] Publicly spoke ({stance}): {new_post}")
                else:
                    self.logger.info(f"  ☕ [{agent['demographics']['name']}] Kept silent (Inner stance: {stance})")

                # --- Faction population statistics ---
                if stance == "support":
                    counts["support"] += 1
                elif stance == "oppose":
                    counts["oppose"] += 1
                else:
                    counts["neutral"] += 1

            except Exception as e:
                self.logger.error(f"Error processing agent {agent_id}: {str(e)}")

        # All agent decisions completed, uniformly refresh public screen
        self.current_posts.update(next_round_posts)

        # ==========================================
        # 4. Calculate core sociological metrics of polarization
        # ==========================================
        total = self.total_agents
        p_support = counts["support"] / total
        p_oppose = counts["oppose"] / total
        p_neutral = counts["neutral"] / total

        # --- Metric 1: Three-state system Shannon entropy ---
        entropy = 0
        for p in [p_support, p_oppose, p_neutral]:
            if p > 0:
                entropy -= p * math.log2(p)

        # --- Metric 2: Bipolarization Index ---
        total_opinions = p_support + p_oppose
        if total_opinions > 0:
            bipolarization = 1.0 - (abs(p_support - p_oppose) / total_opinions)
        else:
            bipolarization = 0.0

        # --- Metric 3: Topic penetration rate (Overall proportion participating in expressing stance) ---
        penetration_rate = total_opinions

        round_data["summary"] = {
            "support_count": counts["support"],
            "oppose_count": counts["oppose"],
            "neutral_count": counts["neutral"],
            # Added proportion output
            "support_ratio": p_support,
            "oppose_ratio": p_oppose,
            "neutral_ratio": p_neutral,
            "system_entropy": entropy,
            "bipolarization_index": bipolarization,
            "penetration_rate": penetration_rate
        }

        # Console concise print
        self.logger.info(
            f"[Round {self.current_round}] Participation: {penetration_rate:.1%} | "
            f"Support: {p_support:.1%} | Oppose: {p_oppose:.1%} | Neutral: {p_neutral:.1%} | "
            f"Entropy: {entropy:.3f} | Bipolarity: {bipolarization:.3f}"
        )

        return round_data

    def save_round_snapshot(self, round_data):
        path = os.path.join(self.results_dir, f"round_{self.current_round}_{self.timestamp}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)

    def save_final_results(self):
        with open(os.path.join(self.results_dir, f"simulation_summary_{self.timestamp}.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(self.simulation_results, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.results_dir, f"post_history_{self.timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.post_history, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. Execution Entry
# ==========================================
if __name__ == "__main__":
    config = {
        "agent_file": "net_ba_100_seed46.json",  # Recommended to use a 100-person network to run polarization
        "model_provider": "qwen",
        "api_key": "your_api_key_here",  # your own KEY
        "model_name": "qwen-turbo",
        "temperature": 0.7,
        "rounds": 15,
        "seed_strategy": "structural",
        "initial_spreader_ratio": 0.1,  # Increased to 10% (5% each for support and oppose) to ensure intense confrontation

        # Two-way copy for controversial topics
        "initial_support": "[Opinion Discussion] Recently, many companies are using AI to automatically write copy, do customer service, and design. I feel that most jobs will eventually be replaced by AI sooner or later.",
        "initial_oppose": "[Opinion Discussion] AI does improve efficiency, but creativity, communication skills, and responsible judgment cannot be separated from humans. Human jobs cannot be completely replaced.",
    }

    simulation = SocialSimulation(config)
    simulation.run_simulation(rounds=config["rounds"])
