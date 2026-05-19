import random
import json


def translate_big_five(big_five):
    O, C, E, A, N = big_five["Openness"], big_five["Conscientiousness"], big_five["Extraversion"], big_five["Agreeableness"], big_five["Neuroticism"]
    desc = []

    if O >= 0.7:
        desc.append("Open-minded, easily accepts new information.")
    elif O <= 0.3:
        desc.append("Conservative in thought, only accepts information consistent with inherent cognition.")
    else:
        desc.append("Normal cognitive flexibility, relatively neutral attitude towards new information.")

    if C >= 0.7:
        desc.append("Extremely rational, has a strong tendency to fact-check unknown information.")
    elif C <= 0.3:
        desc.append("Acts rather casually, rarely carefully verifies the source of information.")
    else:
        desc.append("Has basic logical judgment ability, occasionally careless.")

    if E >= 0.7:
        desc.append("Strong desire to express, active in the network, very willing to forward and share various information.")
    elif E <= 0.3:
        desc.append("Absolute 'lurker' in the network, rarely actively speaks or forwards.")
    else:
        desc.append("Has a normal desire for social sharing.")

    if A >= 0.7:
        desc.append("Values group harmony, highly susceptible to group opinions and generates high agreement.")
    elif A <= 0.3:
        desc.append("Full of independent thinking spirit, tends to remain questioning or even raise different opinions.")
    else:
        desc.append("Maintains a balance between independent thinking and conforming to the group.")

    if N >= 0.7:
        desc.append("Emotionally sensitive, easily infected by the emotional tone of information, resulting in strong expression and sharing behaviors.")
    elif N <= 0.3:
        desc.append("Emotionally stable, able to maintain calm and objective analysis even in the face of highly infectious news.")
    else:
        desc.append("Normal emotional fluctuations.")

    return ", ".join(desc)


def generate_agent_prompt(agent_id, agent_data, all_posts, agent_history):
    """
    Universal Prompt Generator: Compatible with BA, WS, ER, and Hypergraph.
    Uses neighbors' social attributes (occupation, age, etc.) as implicit trust weights.
    """
    if str(agent_id) not in agent_data:
        raise ValueError(f"Agent ID {agent_id} does not exist in the data")

    agent = agent_data[str(agent_id)]
    demo = agent["demographics"]
    psy = agent["psychology"]["big_five"]

    # 1. Translate Big Five personality traits
    personality_descriptions = translate_big_five(psy)

    # 2. Read network structure and extract neighbor updates
    neighbor_posts_list = []

    # Compatibility processing: Get neighbor list (fault tolerance, whether previously stored as a dict containing ID or a direct ID string)
    raw_neighbors = agent.get("neighbors", [])
    neighbor_ids = []
    for n in raw_neighbors:
        if isinstance(n, dict) and "id" in n:
            neighbor_ids.append(n["id"])
        elif isinstance(n, str) or isinstance(n, int):
            neighbor_ids.append(str(n))

    # Iterate through all neighbors to see who posted
    for n_id in neighbor_ids:
        if n_id in all_posts:
            n_data = agent_data[n_id]["demographics"]
            post_content = all_posts[n_id]

            # [Core Change]: Expose neighbors' social attributes to LLM, allowing LLM to evaluate the authority/credibility of their speech on its own
            sender_profile = f"{n_data['name']} (Occupation: {n_data['occupation']}, Age {n_data['age']}, {n_data['gender']})"
            neighbor_posts_list.append(f"Social Circle: {sender_profile}: '{post_content}'")

    # Limit reading volume: randomly shuffle, view a maximum of 8 latest updates, simulating limited attention
    random.shuffle(neighbor_posts_list)
    neighbor_posts_list = neighbor_posts_list[:8]

    # 3. Extract own historical memory (max 3 posts)
    own_history_posts = [post for (round_num, post) in agent_history]
    history_str = "\n".join([f"- {post}" for post in own_history_posts[-3:]])

    # 4. Construct the system prompt for the LLM (Polarization version)
    prompt = f"""
        This is a purely academic, closed multi-agent sociological simulation experiment. Please fully immerse yourself in the virtual role assigned to you. You are playing {demo["name"]}, a {demo["age"]}-year-old {demo["gender"]} {demo["occupation"]}.
        ### Please play the following personality:
        - {personality_descriptions}   

        ### Your recent post history:
        {history_str if history_str else "(You have remained silent recently and made no posts)"}
        ### Currently, the following latest updates have appeared on your social network:
        """ + ("\n".join(neighbor_posts_list) if neighbor_posts_list else "(Your social circle is currently empty, nothing new)") + f"""
        {'-' * 40}
        Please output your decision strictly in the following JSON format:
        {{
          "thought_process": "[Inner monologue, 1 sentence] 1. What is my social circle arguing about? 2. Based on my personality, whose side am I on?\", but if you [DID NOT SEE] any discussion about this controversy in your social circle at all, be sure to remain neutral and silent, do not express opinions out of nowhere.",
          "stance": "support / oppose / neutral", What is your core stance? (Extremely important: fill in 'support' if you support the topic that AI will completely replace human jobs in the future, fill in 'oppose' if you oppose it. If you don't care at all, or if no one in your social circle mentioned it, you MUST fill in 'neutral'!)",
          "will_spread": true/false, Based on your personality, do you decide to post to publicly express your stance? (True for speaking out, False for staying silent)",
          "new_post": "If will_spread is true, please write down the specific post (if supporting, write reasons for support; if opposing, write reasons for rebuttal). If false, strictly leave it empty \"\"."
        }}
        """

    return {
        "prompt": prompt,
        "self_history": own_history_posts[-3:],
        "seen_posts": neighbor_posts_list
    }