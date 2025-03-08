from typing import TypedDict, List
from langchain import hub
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from twilio.rest import Client
import sqlite3
import faiss

load_dotenv()


prompt = hub.pull('rlm/rag-prompt')

llm = ChatOpenAI(model='gpt-4o-mini')

def compute_savings(kg_charcoal_used, charcoal_cost_per_kg=100, 
                    eff_charcoal=0.15, eff_biogas=0.45):
    """
    Compute the energy and cost savings if a household switches from charcoal to biogas.
    
    Assumptions:
      - Charcoal energy density: 29.6 MJ/kg.
      - Biogas energy density: 20 MJ/m³.
      - Charcoal stove efficiency: ~15% (useful energy fraction).
      - Biogas stove efficiency: ~45%.
      - Charcoal cost: ~KES 100 per kg.
      - Biogas is assumed to have no direct cost.
    
    Steps:
      1. Calculate the total energy from the charcoal used.
      2. Calculate the useful energy delivered by charcoal.
      3. Determine the fuel energy required from biogas to deliver the same useful energy.
      4. Convert that required energy (MJ) to the equivalent volume of biogas (m³).
      5. Compute daily and monthly energy and cost savings.
    
    Parameters:
        kg_charcoal_used (float): Kilograms of charcoal used in a day.
        charcoal_cost_per_kg (float): Price per kg of charcoal (default: 100 KES).
        eff_charcoal (float): Efficiency of charcoal stove (default: 0.15).
        eff_biogas (float): Efficiency of biogas stove (default: 0.45).
    
    Returns:
        dict: Contains daily and monthly energy savings (MJ), cost savings (KES),
              and the volume of biogas needed (m³) to deliver the same useful energy.
              Keys: 'daily_energy_savings', 'daily_cost_savings', 
                    'monthly_energy_savings', 'monthly_cost_savings',
                    'biogas_volume_needed'.
    """
    CHARCOAL_ENERGY_DENSITY = 29.6  # MJ per kg
    BIOGAS_ENERGY_DENSITY = 20      # MJ per m³

    # Total energy contained in the charcoal used (MJ)
    energy_from_charcoal = kg_charcoal_used * CHARCOAL_ENERGY_DENSITY

    # Useful energy delivered by charcoal (MJ)
    useful_energy = energy_from_charcoal * eff_charcoal

    # Fuel energy required from biogas to deliver the same useful energy (MJ)
    required_biogas_energy = useful_energy / eff_biogas

    # Convert the required biogas energy to volume (m³)
    biogas_volume_needed = required_biogas_energy / BIOGAS_ENERGY_DENSITY

    # Energy savings (MJ) is the difference between the charcoal fuel energy and
    # the biogas fuel energy required to deliver the same useful energy.
    daily_energy_savings = energy_from_charcoal - required_biogas_energy

    # Daily cost savings (KES) assuming the cost of charcoal is avoided entirely
    daily_cost_savings = kg_charcoal_used * charcoal_cost_per_kg

    # Monthly savings (assuming 30 days in a month)
    monthly_energy_savings = daily_energy_savings * 30
    monthly_cost_savings = daily_cost_savings * 30

    return {
        "daily_energy_savings": daily_energy_savings,
        "daily_cost_savings": daily_cost_savings,
        "monthly_energy_savings": monthly_energy_savings,
        "monthly_cost_savings": monthly_cost_savings,
        "biogas_volume_needed": biogas_volume_needed  # m³ of biogas needed daily
    }


class State(TypedDict):
    kg_charcoal_used: str
    daily_energy_savings: float
    daily_cost_savings: float
    monthly_energy_savings: float
    monthly_cost_savings: float 
    biogas_volume_needed : float
    messages: str


def assistant(state: State):
    print(state['kg_charcoal_used'])
    savings = compute_savings(int(state['kg_charcoal_used']))
    prompt = "You are a computing assistant responsible for calculating energy and cost savings using the provided {savings} data that demonstrates the higher efficiency of biogas compared to charcoal. Please provide a brief explanation in a two-three sentences to the user on why opting for biogas is a more effective choice than charcoal. Please note that prices are computed in Kenya Shillings while energy savings should be in MJ. Make sure all the calculated results are mentioned in your response"
    sys_message = SystemMessage(content=prompt.format(savings=savings))
    response = llm.invoke([sys_message])

    return {
        "daily_energy_savings": savings['daily_energy_savings'],
        "daily_cost_savings": savings['daily_cost_savings'],
        "monthly_energy_savings": savings['monthly_energy_savings'],
        "monthly_cost_savings": savings['monthly_cost_savings'],
        "biogas_volume_needed": savings['biogas_volume_needed'],
        "messages": response # m³ of biogas needed daily
    }    


workflow = StateGraph(State)

# Define nodes: these do the work
workflow.add_node("assistant", assistant)

# Define edges: these determine how the control flow moves
workflow.add_edge(START, "assistant")
workflow.add_edge("assistant", END)
react_graph = workflow.compile()

response = react_graph.invoke({"kg_charcoal_used": "4"})
print(response['messages'].content)
