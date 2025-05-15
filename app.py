import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import requests
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage import segmentation
from lime import lime_image
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
import io
import matplotlib.pyplot as plt
import pandas as pd

# --- Streamlit Page Config ---
st.set_page_config(page_title="Alzheimer’s Detection & Awareness", page_icon="🧠", layout="wide")

# --- Header ---
st.markdown("""
    <style>
        .header {text-align: center; font-size: 32px; font-weight: bold; padding: 10px;}
        .footer {text-align: center; font-size: 16px; margin-top: 50px; padding: 10px;}
        .button-container {text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>🧠 Alzheimer’s Disease Detection & Awareness</div>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload MRI", "🧠 About Alzheimer’s", "📸 MRI Gallery", "💙 Patient Stories", "🧘 Brain Health & Activities", "📊 Affected Area Prediction","🤖 AI Chatbot"])

# --- Define CBAM Attention Module ---
def cbam_block(x, reduction_ratio=16):
    channel = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Dense(channel // reduction_ratio, activation="relu")(avg_pool)
    avg_pool = layers.Dense(channel, activation="sigmoid")(avg_pool)
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Dense(channel // reduction_ratio, activation="relu")(max_pool)
    max_pool = layers.Dense(channel, activation="sigmoid")(max_pool)
    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation("sigmoid")(channel_attention)
    channel_attention = layers.Reshape((1, 1, channel))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
    avg_pool_spatial = layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    x = layers.Multiply()([x, spatial_attention])
    return x

# --- Build Model ---
def build_model():
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    return keras.Model(inputs, outputs)

@st.cache_resource()
def load_model():
    model = build_model()
    model.load_weights("D:\\AlzheimerDisease\\alzheimer_model.weights.h5")
    return model

model = load_model()
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']
# --- Function to Segment Brain Region ---
def get_brain_mask(image_array):
    gray_image = rgb2gray(image_array)  # Convert to grayscale
    thresh = threshold_otsu(gray_image)  # Compute threshold
    binary_mask = gray_image > thresh  # Apply thresholding
    binary_mask = remove_small_objects(binary_mask, min_size=500)  # Remove small non-brain regions
    return binary_mask

# --- Function to Segment Brain Region ---
def get_brain_mask(image_array):
    gray_image = rgb2gray(image_array)  # Convert to grayscale
    thresh = threshold_otsu(gray_image)  # Compute threshold
    binary_mask = gray_image > thresh  # Apply thresholding
    binary_mask = remove_small_objects(binary_mask, min_size=500)  # Remove small non-brain regions
    return binary_mask

# --- LIME Explanation ---
def lime_explanation(img, model):
    original_img = img.copy()  # Save original image for display
    original_size = img.size   # (width, height)

    # Resize image for model input (224x224)
    resized_img = img.resize((224, 224))
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    explainer = lime_image.LimeImageExplainer()

    # Model prediction wrapper
    def model_predict(input_images):
        input_images = np.array(input_images)
        return model.predict(input_images)

    # Get LIME explanation
    explanation = explainer.explain_instance(
        img_array[0], model_predict, top_labels=1, hide_color=0, num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )

    # Restrict explanation to brain region
    brain_mask = get_brain_mask(temp)
    refined_mask = mask * brain_mask
    marked_img = mark_boundaries(temp, refined_mask)

    # Resize marked image to original size using PIL to preserve orientation
    marked_img_pil = Image.fromarray((marked_img * 255).astype(np.uint8))
    marked_img_resized = marked_img_pil.resize(original_size)

    # Display in Streamlit
    st.image(original_img, caption="Uploaded Image", width=700)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(marked_img_resized)
    ax.set_title("LIME Explanation (Scaled to Original Size)", fontsize=10)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf, caption="LIME Explanation", width=700)
    plt.close(fig)



# --- Pages ---
if page == "🏠 Home":
    st.write("Detect Alzheimer’s at different stages using AI-powered MRI analysis and access brain health resources.")
    # Load Image
    img_path = "D:\\AlzheimerDisease\\alzheimer.png"
    img_path2= "D:\\AlzheimerDisease\\alzheimer2.jpg"
    img = Image.open(img_path)
    img2 = Image.open(img_path2)
    small_img1 = img.resize((500, 500))
    small_img2 = img2.resize((700, 500))
    col1, col2 = st.columns(2)
    with col1:
        st.image(small_img1, caption="Image 1", use_column_width=False)

    with col2:
        st.image(small_img2, caption="Image 2", use_column_width=False)
    
elif page == "📤 Upload MRI":
    st.title("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="Uploaded MRI", width=650)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"Prediction: {predicted_class} ({confidence:.2f}%)")

elif page == "🧠 About Alzheimer’s":
    st.title("🧠 Understanding Alzheimer’s Disease")
    
    st.write("""
    **Alzheimer’s disease** is a progressive neurological disorder that affects memory, thinking, and behavior.  
    It is the most common cause of dementia, gradually impairing an individual’s ability to carry out daily activities.  
    Over time, brain cells shrink and die, leading to significant cognitive decline and memory loss.  

    While there is no cure yet, early diagnosis, lifestyle choices, and supportive care can help manage the condition and improve the quality of life.  
    Scientists and researchers worldwide continue to study Alzheimer’s to develop better treatments and, ultimately, find a cure.
    """)

    st.subheader("🧬 **What Causes Alzheimer’s?**")
    st.write("""
    The exact cause of Alzheimer’s is still being studied, but it is believed to result from a combination of genetic, environmental, and lifestyle factors.  
    Key contributors include:
    
    - **Beta-amyloid plaques & Tau tangles** – Abnormal protein deposits in the brain disrupt cell function.
    - **Genetics** – Family history can play a role in increased risk.
    - **Aging** – The risk of Alzheimer’s increases significantly with age.
    - **Lifestyle & Heart Health** – Poor diet, lack of exercise, and chronic conditions (like diabetes and high blood pressure) may contribute.
    """)

    st.subheader("📊 **Stages of Alzheimer’s**")
    st.write("""
    - **Early Stage (Mild Cognitive Impairment)** – Subtle memory lapses, difficulty finding words, and minor confusion.
    - **Middle Stage (Moderate Alzheimer’s)** – Increased memory loss, difficulty recognizing people, trouble performing familiar tasks.
    - **Late Stage (Severe Alzheimer’s)** – Loss of speech, mobility issues, and full dependence on caregivers.
    """)

    st.subheader("💡 **Did You Know?**")
    st.write("""
    - Over **55 million people** worldwide are living with dementia, and Alzheimer's is the leading cause.
    - **Early diagnosis** can help manage symptoms and plan for future care.
    - **Lifestyle changes** like regular exercise, a brain-healthy diet, and staying socially active can reduce the risk.
    """)

    st.subheader("🛠️ **Resources & Support**")
    st.markdown("[Learn More on Wikipedia](https://en.wikipedia.org/wiki/Alzheimer%27s_disease)")
    st.markdown("[Alzheimer's Association](https://www.alz.org/)")
    st.markdown("[National Institute on Aging](https://www.nia.nih.gov/health/alzheimers-disease)")
    st.markdown("[Brain Health Tips](https://www.alz.org/help-support/brain_health)")

    
# 📸 MRI Gallery Page
elif page == "📸 MRI Gallery":
    st.title("📸 MRI Gallery 🧠")

    # List of images and captions
    image_paths = [
        "D:/AlzheimerDisease/Compare.png",
        "D:/AlzheimerDisease/Compare2.png",
        "D:/AlzheimerDisease/compare3.jpeg",
        "D:/AlzheimerDisease/compare4.png",
        "D:/AlzheimerDisease/compare5.jpg"
    ]

    captions = [
        "MRI Comparison: Healthy Control vs Alzheimer's Brain",
        "Structural Changes in Alzheimer's Disease",
        "Brain Atrophy due to Alzheimer's disease",
        "Brain Shrinkage due to Alzheimer disease",
        "Alzheimer's disease affected brain"
    ]

    # Two-column layout
    col1, col2 = st.columns(2)

    # Display images in a grid
    for i in range(0, len(image_paths), 2):
        with col1:
            with st.container():
                st.image(image_paths[i], use_column_width=True, caption=captions[i])
        
        if i + 1 < len(image_paths):
            with col2:
                with st.container():
                    st.image(image_paths[i+1], use_column_width=True, caption=captions[i+1])
                    

    # 📝 Notes Section
    st.markdown("---")
    st.subheader("📝 Notes on MRI Findings")
    st.markdown("""
    - **🧠 Healthy vs Alzheimer's Brain:** Highlights volume loss in an Alzheimer's-affected brain.
    - **📉 Structural Changes:** MRI scans show degeneration, especially in the hippocampus.
    - **🔬 Brain Atrophy:** Progressive shrinkage of brain tissues affects memory and cognition.
    - **📌 Progressive Shrinkage:** As Alzheimer's progresses, noticeable brain matter loss occurs.
    - **🖼️ MRI Patterns:** Different MRI techniques reveal abnormalities in Alzheimer's progression.
    """)



elif page == "💙 Patient Stories":
    st.title("💙 Patient Stories")
    st.write("""
✒️ **A Journey of Strength, Love, and Unbreakable Spirit**  

Alzheimer’s is more than just a diagnosis—it is a journey that tests the very essence of human resilience.  
It is a road paved with both challenges and cherished moments, where memories may fade, but love and determination remain unshaken.  

Behind every MRI scan lies a story—of a life once filled with laughter, of families holding onto precious moments, and of unwavering courage in the face of uncertainty.  
Each individual walking this path teaches us that strength is not measured by the past but by the willingness to embrace each new day with grace.  

Through these stories, we honor those who have faced Alzheimer’s with dignity and perseverance.  
We celebrate the caregivers who devote their hearts to keeping memories alive.  
We embrace the moments of joy found in the simplest of things—a familiar song, the warmth of a loved one’s touch, or the quiet strength of an unspoken bond.  

These stories are not just about loss, but about **love, hope, and the indomitable human spirit.**  
Let them inspire you, remind you of the power of resilience, and fill your heart with warmth.  

📖 *Read the heartfelt journeys of those living with Alzheimer's:*  
""")

    st.markdown("[Real Stories](https://my.clevelandclinic.org/patient-stories/61-slowing-the-onset-of-alzheimers)")

elif page == "🧘 Brain Health & Activities":
    st.title("🧘 Brain Health & Activities")  
    st.write("""  
**🌿 Nurturing the Mind, Enriching the Soul**  

The brain is the seat of our memories, thoughts, and emotions—an intricate masterpiece that thrives on stimulation, care, and balance.  
Just as we nourish our bodies with wholesome foods and movement, our minds too require gentle yet profound engagement to remain sharp and resilient.  

🧩 **Engage & Challenge** – Keep your brain active with puzzles, strategic games, and problem-solving activities. A well-exercised mind is a strong mind!  

📖 **Read & Reflect** – Immerse yourself in the beauty of literature, poetry, and history. Every story, every verse strengthens the mind’s tapestry.  

🎶 **Music & Art** – Whether listening to a melody that tugs at the heart or painting a canvas filled with emotions, creative expression nurtures cognitive well-being.  

🌞 **Nature & Movement** – A quiet walk in the golden glow of sunrise, mindful breathing under a canopy of trees, or gentle yoga flows—nature and movement soothe the soul.  

🧘 **Mindfulness & Meditation** – Amidst the rush of life, stillness is a powerful healer. Practicing mindfulness, meditation, and gratitude fosters mental clarity and emotional strength.  

💬 **Social Connection** – Conversations, laughter, shared memories—human connection is an elixir for the mind. Engage with loved ones, reminisce, and build new experiences together.  

🛠️ **Lifelong Learning** – Curiosity fuels the intellect. Learning a new language, skill, or hobby keeps the neural pathways alive and thriving.  

🫀 **Holistic Well-being** – A well-balanced diet, restful sleep, and a heart filled with joy contribute to lasting brain health. Small daily choices shape the strength of the mind.  

✨ *The mind is a garden; nurture it with care, and it will flourish with wisdom, resilience, and boundless creativity.*  
""")

    st.write("Explore free activities to boost cognitive function:")
    st.markdown("[Lumosity (Brain Games)](https://www.lumosity.com/)")
    st.markdown("[Sudoku Online](https://www.websudoku.com/)")
    st.markdown("[Crossword Puzzles](https://www.nytimes.com/crosswords)")



elif page == "🤖 AI Chatbot":
    import random
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_groq import ChatGroq

    # Hard-coded API key
    GROQ_API_KEY = 'REPLACE_WITH_YOUR_GROQ_API_KEY_HERE'

    # Colors and emojis
    EMOJIS = ["🧠", "🌼", "💬", "🧓", "🩺", "🌟", "❤️", "👵", "🌿"]
    HEADER_COLOR = "#c6e2ff"
    BG_COLOR =  "#4682B4"


    st.markdown(
        f"<h1 style='text-align: center; color: {HEADER_COLOR};'>🧠 Alzheimer Support Chatbot</h1>",
        unsafe_allow_html=True
    )

    st.markdown(f"<div style='background-color: {BG_COLOR}; padding: 15px; border-radius: 10px;'>"
                f"<p style='text-align: center; font-size: 20px; color: white;'>👵 Hi! I'm here to support you with Alzheimer's-related information and care guidance. 💙</p></div>",
                unsafe_allow_html=True)

    # Sidebar settings
    st.sidebar.title('Customize Your Chat 🛠️')
    model = st.sidebar.selectbox('Choose a model:', [
        'llama3-8b-8192', 'llama3-70b-8192', 'gemma-7b-it'
    ])
    conversational_memory_length = st.sidebar.slider('Memory Length (for context):', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Text area for input
    user_question = st.text_area("💬 Ask about Alzheimer’s symptoms, care tips, or any concerns:")

    # Initialize chat history if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Set up Groq Chat
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Process and respond
    if user_question:
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)

        st.markdown(
            f"<div style='background-color: {BG_COLOR}; padding: 10px; border-radius: 10px; margin: 10px;'>"
            f"<p style='color:white'><strong>You {random.choice(EMOJIS)}:</strong> {user_question}</p>"
            f"<p style='color:white'><strong>SupportBot {random.choice(EMOJIS)}:</strong> {response['response']}</p></div>",
            unsafe_allow_html=True
        )

    # Display past conversations
    st.markdown("<h3 style='color: #ffffff;'>🕰️ Previous Conversations</h3>", unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.markdown(
            f"<div style='background-color: {BG_COLOR}; padding: 8px; border-radius: 8px; margin: 8px;'>"
            f"<p style='color:white'><strong>You {random.choice(EMOJIS)}:</strong> {message['human']}</p>"
            f"<p style='color:white'><strong>SupportBot {random.choice(EMOJIS)}:</strong> {message['AI']}</p></div>",
            unsafe_allow_html=True
        )

        
elif page == "📊 Affected Area Prediction":
    st.title("📊 LIME-Based Affected Area Detection")

    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        lime_explanation(img, model)       

    image1 = Image.open("D:\\AlzheimerDisease\\mrilabeled.jpg") 
    image2 = Image.open("D:\AlzheimerDisease\Screenshot 2025-04-12 205919.png") 
    # Display Images Side by Side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image1, caption="MRI brain image labels", use_column_width=True)

    with col2:
        st.image(image2, caption="MRI brain image labels", use_column_width=True)
        # --- Data ---
    data = [
    ["Cingulate Gyrus", "Emotion, behavior regulation, memory", "Often affected early, linked to emotional disturbance and memory", "fMRI, PET", "Cognitive-behavioral therapy"],
    ["Corpus Callosum", "Connects left and right brain", "Thinning observed, affects inter-hemispheric communication", "MRI", "Memory exercises, inter-hemispheric coordination tasks"],
    ["Fornix", "Memory processing (connects hippocampus to other areas)", "Atrophy leads to memory impairment", "Diffusion MRI", "Memory training, cholinesterase inhibitors"],
    ["Caudate Nucleus", "Motor control, learning, memory", "Affected in advanced dementia, linked to motor decline", "MRI, PET", "Physical therapy"],
    ["Putamen", "Motor skills, learning", "Can show volume loss in late stages", "MRI", "Motor rehabilitation"],
    ["Thalamus", "Sensory relay, consciousness", "Involved in cognition, atrophies in late Alzheimer’s", "fMRI, DTI", "Mental stimulation"],
    ["Hypothalamus", "Hormone control, circadian rhythms", "Can affect sleep cycles and appetite in dementia", "Sleep studies, MRI", "Sleep therapy, melatonin regulation"],
    ["Pineal Gland", "Regulates sleep (melatonin)", "Disrupted in dementia → sleep issues", "Hormone levels, MRI", "Melatonin supplements"],
    ["Superior Colliculus", "Visual reflexes", "Rarely affected but visual misperceptions can occur", "Visual tracking", "Visual cues, lighting therapy"],
    ["Inferior Colliculus", "Auditory processing", "May contribute to difficulty processing speech", "Auditory tests", "Speech therapy"],
    ["Cerebral Aqueduct", "CSF flow", "Can enlarge in hydrocephalus (pressure-related dementia)", "MRI, CSF tap test", "Monitor CSF pressure, shunt in NPH cases"],
    ["Fourth Ventricle", "CSF circulation", "Enlargement seen with brain atrophy", "MRI", "Hydration, posture, pressure management"],
    ["Cerebellum", "Balance, coordination", "Affected in some dementias (ataxia symptoms)", "Balance tests, MRI", "Physical therapy"],
    ["Temporal Lobe", "Hearing, memory, speech", "Highly involved in early Alzheimer’s", "Cognitive tests, MRI", "Memory therapy, speech therapy"],
    ["Lateral Ventricles (incl. Inferior Horn)", "CSF flow", "Enlargement due to cortical atrophy", "MRI", "Monitor for NPH"],
    ["Third Ventricle", "CSF pathway", "Enlargement in progressive dementia", "MRI", "Same as above"],
    ["Lateral Geniculate Nucleus (LGN)", "Visual processing", "Not directly affected, but visual hallucinations may occur", "Visual field tests", "Address lighting and visual stimulation"],
    ["Superior Temporal Gyrus", "Auditory/language processing", "Degenerates → word-finding difficulties", "Verbal fluency test", "Language-based therapy"],
    ["Inferior Temporal Gyrus", "Visual recognition", "Atrophy linked to visual agnosia (can’t recognize faces/objects)", "Object recognition tasks", "Supportive labeling"],
    ["Superior Longitudinal Fissure", "Separates hemispheres", "Anatomical marker only; not directly affected", "-", "-"],
    ["Occipital Lobe", "Visual processing", "Affected in posterior cortical atrophy variant", "Visual-spatial tests", "Visual aids, contrast-rich environments"]
]

# Create DataFrame
    df = pd.DataFrame(data, columns=[
    "Region",
    "Function",
    "Relevance to Alzheimer’s/Dementia",
    "Tests",
    "Remedies / Support"
])

# --- UI ---

    st.title("🧠 Alzheimer's Brain Mapping Tool")
    st.markdown("### Explore the effects of Alzheimer's on different brain regions using MRI and related imaging techniques.")

    st.subheader("🧠 Comprehensive Brain Region Breakdown")
    st.dataframe(df, use_container_width=True)

    st.markdown("""
> To determine whether Alzheimer's disease has affected the brain using MRI images, radiologists and neurologists focus on specific brain regions.  
> These are highlighted in:
- **"Neuroimaging in Dementia" by Piersol and Dickerson**  
- **"Magnetic Resonance Imaging of the Brain and Spine" by Scott W. Atlas**  
- **"Principles of Neural Science" by Eric Kandel et al.**
""")

# --- Alzheimer’s Key Indicators ---
    st.subheader("🔹 1. Hippocampus")
    st.markdown("""
- **Primary region affected in early-stage Alzheimer’s disease.**  
- Significant **atrophy (shrinkage)** is a hallmark of the disease.  
- Responsible for **memory formation**, and its deterioration leads to memory loss.
""")

    st.subheader("🔹 2. Medial Temporal Lobe")
    st.markdown("""
- Includes **entorhinal cortex, amygdala, and parahippocampal gyrus**.  
- Early-stage Alzheimer’s **destroys neurons in these regions**, causing difficulty in forming new memories.
""")

    st.subheader("🔹 3. Ventricular Enlargement (Hydrocephalus ex Vacuo)")
    st.markdown("""
- As brain tissue shrinks, **lateral ventricles expand** to compensate.  
- Enlargement is a sign of **severe neurodegeneration**.
""")

    st.subheader("🔹 4. Posterior Cingulate Cortex & Precuneus")
    st.markdown("""
- These show **early metabolic decline**, detectable via MRI and PET.  
- fMRI reveals **reduced activity** even before visible damage.
""")

    st.subheader("🔹 5. Cortical Atrophy (Grey Matter Loss)")
    st.markdown("""
- **Parietal and temporal lobes** most affected mid-to-late stages.  
- **Frontal lobe** degeneration in advanced stages → cognitive decline and behavior changes.
""")

    st.subheader("🔹 6. White Matter Hyperintensities (WMH)")
    st.markdown("""
- Seen in **T2-weighted and FLAIR MRI**.  
- Indicate **vascular damage**, common in Alzheimer’s and vascular dementia.
""")

    st.subheader("🔹 7. Amyloid and Tau Deposits (Advanced Imaging Techniques)")
    st.markdown("""
- **Amyloid plaques** and **tau tangles** are core features of Alzheimer’s.  
- Confirmed with **Amyloid PET** and **Tau PET** scans, complementing MRI results.
""")
st.markdown("<div class='footer'>© 2025 Alzheimer’s Awareness Project</div>", unsafe_allow_html=True)
