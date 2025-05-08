import streamlit as st
import os
from TravelAgents import guide_expert, location_expert, planner_expert, set_openai_api_key
from TravelTasks import location_task, guide_task, planner_task
from crewai import Crew, Process

# Streamlit App Title
st.title("🌍 AI-Powered Trip Planner")

# Step 1: Ask for OpenAI API key
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if not st.session_state.api_key_set:
    openai_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
    if st.button("✅ Submit API Key"):
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            set_openai_api_key(openai_key)
            st.session_state.api_key_set = True
            st.success("API key saved! You can now use the app.")
            st.rerun()
        else:
            st.error("Please enter a valid OpenAI API key.")
else:
    # Show the rest of the app after API key is set
    st.markdown("""
    💡 **Plan your next trip with AI!**  
    Enter your travel details below, and our AI-powered travel assistant will create a personalized itinerary including:
    Best places to visit 🎡   Accommodation & budget planning 💰  
    Local food recommendations 🍕   Transportation & visa details 🚆
    """)

    from_city = st.text_input("🏡 From City", "India")
    destination_city = st.text_input("✈️ Destination City", "Rome")
    date_from = st.date_input("📅 Departure Date")
    date_to = st.date_input("📅 Return Date")
    interests = st.text_area("🎯 Your Interests (e.g., sightseeing, food, adventure)", "sightseeing and good food")

    if st.button("🚀 Generate Travel Plan"):
        if not from_city or not destination_city or not date_from or not date_to or not interests:
            st.error("⚠️ Please fill in all fields before generating your travel plan.")
        else:
            st.write("⏳ AI is preparing your personalized travel itinerary... Please wait.")

            loc_task = location_task(location_expert, from_city, destination_city, date_from, date_to)
            guid_task = guide_task(guide_expert, destination_city, interests, date_from, date_to)
            plan_task = planner_task([loc_task, guid_task], planner_expert, destination_city, interests, date_from, date_to)

            crew = Crew(
                agents=[location_expert, guide_expert, planner_expert],
                tasks=[loc_task, guid_task, plan_task],
                process=Process.sequential,
                full_output=True,
                verbose=True,
            )

            result = crew.kickoff()

            st.subheader("✅ Your AI-Powered Travel Plan")
            st.markdown(result)

            travel_plan_text = str(result)
            st.download_button(
                label="📥 Download Travel Plan",
                data=travel_plan_text,
                file_name=f"Travel_Plan_{destination_city}.txt",
                mime="text/plain"
            )
