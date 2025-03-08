import streamlit as st
from graph import react_graph  # Import the compiled workflow from workflow.py

st.title("Biogas vs. Charcoal Savings Calculator")

# Get user input for the kilograms of charcoal used in a day
kg_charcoal = st.text_input("Enter the kilograms of charcoal used in a day:", "4")

if st.button("Compute Savings"):
    try:
        # Invoke the workflow with the user's input
        result = react_graph.invoke({"kg_charcoal_used": kg_charcoal})

        # Display the computed savings
        st.subheader("Biogas Savings")
        st.write(f"**Biogas Volume Needed Daily (m³):** {result['biogas_volume_needed']:.2f}")
        st.write(f"**Daily Energy Savings (MJ):** {result['daily_energy_savings']:.2f}")
        st.write(f"**Daily Cost Savings (KES):** {result['daily_cost_savings']:.2f}")
        st.write(f"**Monthly Energy Savings (MJ):** {result['monthly_energy_savings']:.2f}")
        st.write(f"**Monthly Cost Savings (KES):** {result['monthly_cost_savings']:.2f}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
