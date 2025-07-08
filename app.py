import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from tae_data import TAE_UNITS_DATA # Import our TAE data
import json # Import the json module for json.dumps()

        # --- Configuration ---
        load_dotenv() 
        API_KEY = os.getenv("GEMINI_API_KEY")

        if not API_KEY:
            st.error("GEMINI_API_KEY not found in .env file. Please set it up as per instructions.")
            st.stop()

        genai.configure(api_key=API_KEY)

        model = genai.GenerativeModel('gemini-1.5-flash') 

        # --- Streamlit App Layout Configuration ---
        st.set_page_config(
            page_title="RPL AI Assistant for Cert IV TAE",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("RPL AI Assistant for Certificate IV in Training and Assessment (TAE40122)")
        st.markdown("---") 

        # --- Session State Management ---
        if 'stage' not in st.session_state:
            st.session_state.stage = 0 # 0: Welcome, 1: Experience Input, 1_post_analysis: Show AI suggestions + Mock Evidence Input, 2: Mapping Results
        if 'user_experience' not in st.session_state:
            st.session_state.user_experience = ""
        if 'evidence_snippets' not in st.session_state:
            st.session_state.evidence_snippets = {}
            for key in ["training_plan", "participant_feedback", "learning_resources", 
                        "assessment_tool", "assessment_records", 
                        "desc_training_session", "desc_assessment_process"]:
                st.session_state.evidence_snippets[key] = ""
        if 'mapping_results' not in st.session_state:
            st.session_state.mapping_results = None
        if 'ai_relevant_units_assessment' not in st.session_state:
            st.session_state.ai_relevant_units_assessment = [] 
        if 'ai_initial_analysis_done' not in st.session_state: # New flag to track if initial AI analysis has happened
            st.session_state.ai_initial_analysis_done = False

        # --- Functions for AI Interaction with Gemini API ---

        def analyze_experience_with_gemini(experience_text):
            """
            Sends the user's experience summary to Gemini to identify relevant TAE40122 core units
            and suggest evidence types.
            """
            prompt = f"""\
        You are a VET assessor for TAE40122 Cert IV in Training and Assessment.
        Candidate experience: "{experience_text}"

        Identify relevance to these TAE40122 core units:
        - TAEDEL411 Facilitate vocational training
        - TAEASS412 Assess competence

        For each unit, state relevance ("Strongly Relevant" | "Moderately Relevant" | "Not Clearly Relevant") and list 3-5 specific evidence types.

        Output JSON:
        {{
            "relevant_units_assessment": [
                {{
                    "unit": "TAEDEL411 Facilitate vocational training",
                    "relevance": "Strongly Relevant",
                    "suggestions": ["Training plans", "Participant feedback", "Learning resources"]
                }},
                {{
                    "unit": "TAEASS412 Assess competence",
                    "relevance": "Moderately Relevant",
                    "suggestions": ["Assessment tools", "Assessment records", "Feedback to candidates"]
                }}
            ]
        }}
        """
            try:
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                return response.text
            except Exception as e:
                st.error(f"Error calling Gemini API for experience analysis: {e}")
                return None

        def map_evidence_with_gemini(evidence_snippet, unit_code, pc_description):
            """
            Sends an evidence snippet (text from a text area) and a specific Performance Criterion (PC)
            to Gemini for mapping and confidence assessment.
            """
            # FIX: Use json.dumps() to safely embed evidence_snippet, handling internal quotes
            prompt = f"""\
        You are a VET assessor for TAE40122.
        Assess evidence against a Performance Criterion (PC).

        Unit: {unit_code}
        PC: {pc_description}

        Candidate's Evidence: {json.dumps(evidence_snippet)}

        Output JSON:
        {{
            "confidence": "High" | "Medium" | "Low" | "None",
            "explanation": "Concise reason for confidence, referencing evidence.",
            "suggested_action": "Actionable advice for gaps."
        }}

        Confidence levels: High (strong support), Medium (partial/implies, needs more), Low/None (no clear support, missing).
        """
            try:
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                return response.text
            except Exception as e:
                st.warning(f"Error mapping evidence for '{unit_code}' - PC '{pc_description}' (snippet: '{evidence_snippet[:100]}...'): {e}. Skipping this mapping.")
                return None


        def process_all_mock_evidence():
            """
            Orchestrates the process of sending mock evidence snippets to Gemini
            for mapping against PCs. This function now explicitly moves to stage 2.
            Adds debugging outputs.
            """
            all_mapping_results = {}
            target_units = ["TAEDEL411 Facilitate vocational training", "TAEASS412 Assess competence"]
            
            mock_evidence_corpus = "\n\n--- EVIDENCE SECTION ---\n\n".join(
                [f"Type: {label}\nContent: {text}" for label, text in st.session_state.evidence_snippets.items() if text.strip()]
            )

            # --- DEBUGGING START ---
            st.info(f"DEBUG: Combined Mock Evidence Corpus length: {len(mock_evidence_corpus)}. Starts with: '{mock_evidence_corpus[:200]}...'")
            if not mock_evidence_corpus.strip():
                st.warning("WARNING: No readable mock evidence snippets provided. The corpus is empty or whitespace only.")
                st.session_state.mapping_results = {} 
                st.session_state.stage = 2 
                return
            # --- DEBUGGING END ---

            with st.spinner("Analyzing your mock evidence and building your personalized map... This may take a moment."):
                for unit_code, unit_info in TAE_UNITS_DATA.items():
                    if unit_code not in target_units: 
                        continue

                    all_mapping_results[unit_code] = {}
                    for pc_key, pc_desc in unit_info["elements"].items():
                        mapping_json_str = map_evidence_with_gemini(mock_evidence_corpus, unit_code, pc_desc)
                        
                        # --- FIX ADDED HERE (from previous step) ---
                        # Check if the API call failed and returned None
                        if mapping_json_str is None: 
                            st.warning(f"DEBUG: Gemini call for {pc_key} ({unit_code}) returned None. Skipping JSON parsing.")
                            all_mapping_results[unit_code][pc_key] = {
                                "confidence": "None",
                                "explanation": "Gemini API call failed for this criterion.",
                                "suggested_action": "Check API key, internet connection, or try again."
                            }
                            continue # Skip to the next PC
                        # --- END FIX ---

                        # DEBUGGING: This line now runs ONLY if mapping_json_str is NOT None
                        st.info(f"DEBUG: Gemini raw response for {pc_key} ({unit_code}): {mapping_json_str[:300]}...") 

                        if mapping_json_str: # This check is still useful for JSONDecodeError, but `is None` check is more specific
                            import json
                            try:
                                mapping_data = json.loads(mapping_json_str)
                                all_mapping_results[unit_code][pc_key] = mapping_data
                            except json.JSONDecodeError:
                                st.error(f"ERROR: Could not parse AI response for {unit_code} - PC {pc_key}. Raw: {mapping_json_str[:500]}...") # Show more of raw for error
                                all_mapping_results[unit_code][pc_key] = {
                                    "confidence": "Low",
                                    "explanation": "AI response parsing failed. Review evidence manually.",
                                    "suggested_action": "Ensure clarity and relevance of evidence to this criterion."
                                }
                        # Else condition for if mapping_json_str was None is now handled by the `if mapping_json_str is None:` block above
                
            st.session_state.mapping_results = all_mapping_results
            st.session_state.stage = 2 

        # --- UI Logic based on App Stage ---

        # Stage 0: Welcome and Introduction
        if st.session_state.stage == 0:
            st.header("Welcome!")
            st.write("This tool is designed to help you understand what evidence you might need for your Recognition of Prior Learning (RPL) application for the Cert IV in Training and Assessment (TAE40122).")
            st.write("""
            **How it works:**
            1.  Tell us briefly about your current or past training and assessment experience.
            2.  Our AI will suggest relevant units and types of evidence.
            3.  You can then provide **mock examples of your evidence** by typing/pasting text snippets.
            4.  The AI will analyze the text and show you how your evidence *might* map to specific criteria and highlight any gaps.
            """)
            if st.button("Start Your RPL Journey"):
                st.session_state.stage = 1
                st.rerun() 

        # Stage 1: Experience Input and Initial AI Analysis Display
        elif st.session_state.stage == 1:
            st.header("Tell Us About Your Experience")
            st.write("Please provide a brief summary (2-3 sentences, or a short paragraph) of your professional experience related to training and assessment.")
            
            user_input = st.text_area(
                "Your Experience Summary:",
                value=st.session_state.user_experience, 
                height=100,
                placeholder='Example: "I have delivered workplace training to new employees on our operational procedures for the last 3 years. I\'ve also developed some basic training materials for these sessions. I often assess their skills on-the-job."'
            )
            st.session_state.user_experience = user_input 

            # This button triggers the AI analysis of experience
            if st.button("Analyze My Experience & Get Suggestions"):
                if user_input.strip(): # Check if input is not empty
                    with st.spinner("Analyzing your experience..."):
                        ai_analysis_json_str = analyze_experience_with_gemini(user_input)
                        if ai_analysis_json_str:
                            import json
                            try:
                                ai_analysis = json.loads(ai_analysis_json_str)
                                st.session_state.ai_relevant_units_assessment = ai_analysis.get("relevant_units_assessment", [])
                                st.session_state.ai_initial_analysis_done = True # Set flag to true
                                st.rerun() # Rerun to display suggestions and mock evidence inputs
                            except json.JSONDecodeError:
                                st.error("AI's initial analysis response was not valid JSON. Please try again or refine your input.")
                        else:
                            st.error("Could not get a response from the AI for experience analysis.")
                else:
                    st.warning("Please enter your experience summary to proceed.")

        # Stage 1_post_analysis: Display AI suggestions AND Mock Evidence Input
        # This stage is only entered if stage is 1 AND initial analysis has been done
        if st.session_state.stage == 1 and st.session_state.ai_initial_analysis_done:
            st.subheader("AI's Initial Interpretation:")
            if st.session_state.ai_relevant_units_assessment:
                st.write(f"Based on your input, here's how your experience relates to the core TAE40122 units:")
                for unit_data in st.session_state.ai_relevant_units_assessment:
                    st.write(f"- **{unit_data['unit']}**: *{unit_data['relevance']}*")

                st.markdown("**To help us further map your experience, please consider providing the following types of evidence for these units:**")
                for unit_data in st.session_state.ai_relevant_units_assessment:
                    if unit_data['suggestions']:
                        st.write(f"**For {unit_data['unit']}:**")
                        for sug in unit_data['suggestions']:
                            st.write(f"- {sug}")
                    else:
                        st.write(f"**For {unit_data['unit']}:** No specific suggestions provided by AI based on relevance.")
            else:
                st.warning("AI could not identify relevant units or suggestions based on your input. Please provide more detail in the previous step.")

            st.write("---")
            st.subheader("Now, provide your mock evidence!")
            st.write("Paste or type a brief description or snippet for each type of evidence you might have. (This simulates AI analysis of real documents).")

            # Mock evidence input fields
            evidence_keys = {
                "training_plan": "Training Plan / Session Outline Snippet",
                "participant_feedback": "Participant Feedback / Evaluations Snippet",
                "learning_resources": "Learning Resources (e.g., presentations, handouts) Snippet",
                "assessment_tool": "Assessment Tool / Checklist Snippet",
                "assessment_records": "Records of Assessment Decisions / Feedback Snippet",
                "desc_training_session": "Description of a Training Session You Facilitated Snippet",
                "desc_assessment_process": "Description of an Assessment Process You Conducted Snippet",
            }
            # No st.form here to allow typing without immediate reruns from each text area
            for key, label in evidence_keys.items():
                st.session_state.evidence_snippets[key] = st.text_area(
                    label,
                    value=st.session_state.evidence_snippets.get(key, ""), 
                    height=70,
                    key=f"evidence_{key}_input" # Unique key for each text area
                )
            
            # Button to trigger analysis of mock evidence. This will call the processing function.
            if st.button("Analyze My Mock Evidence & Build Map"):
                process_all_mock_evidence() 

        # Stage 2: Display Personalized RPL Mapping Report
        elif st.session_state.stage == 2:
            st.header("Your Personalized RPL Mapping Report (Prototype View)")
            st.write("""
            Here's how your provided mock evidence *could* map to the core units: **TAEDEL411 Facilitate vocational training** and **TAEASS412 Assess competence**.
            """)
            st.markdown("""
            * **✔️ High:** Strong indication of evidence for this criterion.
            * **❓ Medium:** Some evidence, but might need more detail or clarification.
            * **❌ None/Low:** No clear evidence identified yet for this criterion, or evidence is weak.
            """)

            # Display mapping for each target unit
            target_units = ["TAEDEL411 Facilitate vocational training", "TAEASS412 Assess competence"]

            for unit_code_full_name in target_units:
                st.subheader(f"Unit: {unit_code_full_name}")
                if unit_code_full_name in st.session_state.mapping_results:
                    unit_mapping = st.session_state.mapping_results[unit_code_full_name]
                    
                    table_data = []
                    for pc_key, pc_desc in TAE_UNITS_DATA[unit_code_full_name]["elements"].items():
                        result = unit_mapping.get(pc_key, {
                            "confidence": "None",
                            "explanation": "No AI analysis performed for this PC, or a problem occurred.",
                            "suggested_action": "Please ensure the unit is correctly defined or provide relevant evidence."
                        })

                        icon = ""
                        if result["confidence"] == "High":
                            icon = "✔️"
                        elif result["confidence"] == "Medium":
                            icon = "❓"
                        else: 
                            icon = "❌"

                        table_data.append({
                            "Performance Criterion": f"{pc_key} {pc_desc}",
                            "Status": f"{icon} {result['confidence']}",
                            "AI Explanation": result["explanation"],
                            "AI Suggested Actions / Gaps": result["suggested_action"]
                        })
                    
                    st.table(table_data)
                else:
                    st.info(f"No detailed mapping results found for {unit_code_full_name}.")
            
            if st.button("Start Over"):
                st.session_state.stage = 0
                st.session_state.user_experience = ""
                st.session_state.evidence_snippets = {} 
                st.session_state.mapping_results = None
                st.session_state.ai_relevant_units_assessment = []
                st.session_state.ai_initial_analysis_done = False # Reset this flag too
                st.rerun() 

