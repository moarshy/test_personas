import os
import yaml
import random
import openai
import logging
from pathlib import Path
from typing import List, Dict
from naptha_sdk.schemas import AgentRunInput


logger = logging.getLogger(__name__)


async def load_personas(personas_dir: str, num_personas: int) -> List[Dict]:
    """Load random personas from the personas directory"""
    logger.info(f"Loading {num_personas} personas from {personas_dir}")
    personas = []
    personas_path = Path(personas_dir)
    yaml_files = list(personas_path.glob('*.yaml'))
    
    logger.info(f"Found {len(yaml_files)} persona files")
    
    # Randomly select num_personas files
    selected_files = random.sample(yaml_files, min(num_personas, len(yaml_files)))
    
    for file_path in selected_files:
        logger.debug(f"Loading persona from {file_path}")
        try:
            with open(file_path, 'r') as f:
                persona = yaml.safe_load(f)
                personas.append(persona)
                logger.debug(f"Loaded persona: {persona['name']}")
        except Exception as e:
            logger.error(f"Error loading persona from {file_path}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(personas)} personas")
    return personas

def format_persona_for_prompt(persona: Dict) -> str:
    """Format persona details into a clear prompt format"""
    logger.debug(f"Formatting prompt for persona: {persona['name']}")
    return f"""You are {persona['name']}.

Background and Personality:
{persona['persona']}

Your objectives are:
{' '.join('- ' + obj for obj in persona['objectives'])}

You are a {persona['role']} with the following trader characteristics:
{', '.join(persona['trader_type'])}

Please analyze the question from your unique perspective, considering your background, objectives, and personality traits."""

async def get_individual_response(client, persona: Dict, question: str) -> str:
    """Get response from a single persona"""
    logger.info(f"Getting individual response from persona: {persona['name']}")
    
    system_prompt = """You are a market participant with specific personality traits, background, and objectives. 
Analyze the given question from your persona's perspective, considering their unique characteristics, experience, and goals.
Provide a detailed response that reflects your persona's viewpoint, knowledge level, and decision-making style."""

    user_prompt = f"""
{format_persona_for_prompt(persona)}

Question: {question}

Provide your analysis and opinion on this question, staying true to your persona's characteristics."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        logger.debug(f"Received response for {persona['name']}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting response for {persona['name']}: {str(e)}")
        raise

async def get_collective_response(client, persona: Dict, question: str, other_responses: List[str]) -> str:
    """Get response from a persona after seeing other responses"""
    logger.info(f"Getting collective response from persona: {persona['name']}")
    
    system_prompt = """You are a market participant in a group discussion. 
Consider both your unique perspective and the insights shared by others when forming your final opinion.
Maintain your persona's characteristics while engaging with others' viewpoints."""

    user_prompt = f"""
{format_persona_for_prompt(persona)}

Question: {question}

Other participants have shared these perspectives:
{'-' + '\n-'.join(other_responses)}

Considering these viewpoints and your own characteristics, provide your final analysis and opinion."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6
        )
        logger.debug(f"Received collective response for {persona['name']}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting collective response for {persona['name']}: {str(e)}")
        raise

async def run(agent_run: AgentRunInput, *args, **kwargs):
    """Main run function for the agent"""
    logger.info("Starting persona simulation run")
    logger.info(f"Question: {agent_run.inputs.question}")
    logger.info(f"Number of personas requested: {agent_run.inputs.num_personas}")
    
    client = openai.AsyncClient()
    
    # Load personas
    personas_dir = kwargs.get('agents_dir', './market_agents_personas')
    if 'market_agents_personas' not in personas_dir:
        personas_dir = os.path.join(personas_dir, 'personas/market_agents_personas')
    personas = await load_personas(personas_dir, agent_run.inputs.num_personas)
    
    # First iteration: Individual responses
    logger.info("Starting first iteration: Individual responses")
    individual_responses = {}
    for persona in personas:
        response = await get_individual_response(client, persona, agent_run.inputs.question)
        individual_responses[persona['name']] = response
    logger.info(f"Completed individual responses for {len(individual_responses)} personas")
    
    # Second iteration: Collective responses
    logger.info("Starting second iteration: Collective responses")
    collective_responses = {}
    for persona in personas:
        # Get all other responses except current persona's
        other_responses = [resp for name, resp in individual_responses.items() 
                         if name != persona['name']]
        
        response = await get_collective_response(
            client, 
            persona, 
            agent_run.inputs.question,
            other_responses
        )
        collective_responses[persona['name']] = response
    logger.info(f"Completed collective responses for {len(collective_responses)} personas")
    
    # Format final output
    result = {
        "individual_responses": individual_responses,
        "collective_responses": collective_responses
    }
    
    logger.info("Completed persona simulation run")
    return result