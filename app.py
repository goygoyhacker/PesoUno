# app.py - PisoUno Complete Coin Detection System
# Old and New 1 Peso Coin Detection, Classification, Counting, and Total Value

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Import required packages
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("OpenCV not loaded. Please check installation.")

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn not loaded. Please check installation.")

if not CV2_AVAILABLE or not SKLEARN_AVAILABLE:
    st.stop()

# ============================================
# Feature Extraction Functions
# ============================================

def extract_coin_features(image, img_size=(64, 64)):
    """
    Extract features from a coin image for classification
    Returns combined feature vector
    """
    try:
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_resized = cv2.resize(image, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Hu Moments (shape features - 7 features)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 2. Color Histograms (3 channels x 8 bins = 24 features)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        # 3. Edge Density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])
        
        # 4. Texture features
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        
        # Combine all features
        features = np.hstack([hu_moments, hist_features, [edge_density, mean_intensity, std_intensity]])
        
        # Ensure consistent feature size (pad if needed)
        target_size = 100
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
        
    except Exception as e:
        return None

def detect_coins_in_image(image):
    """
    Detect all coins in an image using Hough Circle Transform
    Returns list of detected coin regions
    """
    try:
        # Convert to OpenCV format
        if isinstance(image, Image.Image):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Hough Circle Detection
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=100
        )
        
        coin_regions = []
        annotated_img = img.copy()
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Extract coin region
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(img.shape[1], x + r)
                y2 = min(img.shape[0], y + r)
                
                coin_roi = img[y1:y2, x1:x2]
                
                if coin_roi.size > 0:
                    coin_regions.append({
                        'roi': coin_roi,
                        'position': (x, y),
                        'radius': r,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    # Draw circle on annotated image
                    cv2.circle(annotated_img, (x, y), r, (0, 255, 0), 2)
        
        return coin_regions, annotated_img
        
    except Exception as e:
        return [], None

# ============================================
# Training Function
# ============================================

def train_classifier_from_uploads(old_images, new_images):
    """
    Train SVM classifier from uploaded images
    """
    X = []
    y = []
    
    # Process old coins (label 0)
    for img in old_images:
        features = extract_coin_features(img)
        if features is not None:
            X.append(features)
            y.append(0)
    
    # Process new coins (label 1)
    for img in new_images:
        features = extract_coin_features(img)
        if features is not None:
            X.append(features)
            y.append(1)
    
    if len(X) < 4:
        return None, None, 0
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM
    classifier = SVC(kernel='rbf', probability=True, random_state=42)
    classifier.fit(X_scaled, y)
    
    # Calculate training accuracy
    accuracy = classifier.score(X_scaled, y)
    
    return classifier, scaler, accuracy

# ============================================
# Classification Function
# ============================================

def classify_coin(coin_roi, classifier, scaler):
    """
    Classify a single coin as OLD or NEW
    """
    try:
        features = extract_coin_features(coin_roi)
        if features is None:
            return "UNKNOWN", 0.0
        
        features_scaled = scaler.transform([features])
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        coin_type = "NEW" if prediction == 1 else "OLD"
        return coin_type, confidence
        
    except Exception as e:
        return "UNKNOWN", 0.0

# ============================================
# Streamlit UI
# ============================================

st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .coin-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .old-coin {
        color: #8B4513;
        font-size: 2rem;
        font-weight: bold;
    }
    .new-coin {
        color: #4169E1;
        font-size: 2rem;
        font-weight: bold;
    }
    .total-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298, #1e3c72);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>PisoUno</h1>
    <p>Old and New 1 Peso Coin Detection, Classification, and Counting System</p>
    <p>Denomination: 1 Peso | Value per coin: PHP 1.00</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.scaler = None
    st.session_state.trained = False
    st.session_state.total_old = 0
    st.session_state.total_new = 0
    st.session_state.total_value = 0

# Sidebar for training
with st.sidebar:
    st.header("Training Phase")
    st.markdown("Upload sample images to train the classifier")
    
    st.subheader("Upload OLD 1 Peso Coins")
    old_train = st.file_uploader(
        "Select OLD coin images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="old_train"
    )
    
    st.subheader("Upload NEW 1 Peso Coins")
    new_train = st.file_uploader(
        "Select NEW coin images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="new_train"
    )
    
    if st.button("Train Classifier", type="primary"):
        if old_train and new_train:
            if len(old_train) >= 2 and len(new_train) >= 2:
                with st.spinner("Training classifier. Please wait..."):
                    # Load images
                    old_imgs = [Image.open(f) for f in old_train]
                    new_imgs = [Image.open(f) for f in new_train]
                    
                    # Train
                    classifier, scaler, accuracy = train_classifier_from_uploads(old_imgs, new_imgs)
                    
                    if classifier is not None:
                        st.session_state.classifier = classifier
                        st.session_state.scaler = scaler
                        st.session_state.trained = True
                        
                        st.success(f"Training completed successfully!")
                        st.metric("Training Accuracy", f"{accuracy:.1%}")
                        st.info(f"Trained on {len(old_imgs)} OLD and {len(new_imgs)} NEW coins")
                    else:
                        st.error("Training failed. Please ensure images are clear.")
            else:
                st.warning("Please upload at least 2 images per coin type")
        else:
            st.warning("Please upload both OLD and NEW coin images")
    
    if st.session_state.trained:
        st.divider()
        st.success("Classifier Ready")
        
        # Reset counters button
        if st.button("Reset Counters"):
            st.session_state.total_old = 0
            st.session_state.total_new = 0
            st.session_state.total_value = 0
            st.rerun()

# Main area
st.header("Detection Phase")

if not st.session_state.trained:
    st.warning("Please train the classifier first using the sidebar")
    st.info("Upload OLD and NEW 1 Peso coin images, then click 'Train Classifier'")
    
    with st.expander("How to use this system"):
        st.markdown("""
        **Step 1: Training**
        - Upload at least 2 images of OLD 1 Peso coins
        - Upload at least 2 images of NEW 1 Peso coins
        - Click 'Train Classifier'
        
        **Step 2: Detection**
        - Upload an image containing 1 Peso coins
        - The system will detect all coins and classify them
        - Results show counts and total value
        
        **Note:** For best results, use clear, well-lit images with coins on a plain background.
        """)
else:
    # Display current totals
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="coin-card">
            <div style="font-size: 1.2rem;">OLD 1 Peso Coins</div>
            <div class="old-coin">{st.session_state.total_old}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="coin-card">
            <div style="font-size: 1.2rem;">NEW 1 Peso Coins</div>
            <div class="new-coin">{st.session_state.total_new}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="coin-card">
            <div style="font-size: 1.2rem;">Total Value</div>
            <div class="total-value">PHP {st.session_state.total_value}.00</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Test image upload
    test_file = st.file_uploader(
        "Upload coin image for detection",
        type=['jpg', 'jpeg', 'png'],
        help="Image can contain multiple 1 Peso coins"
    )
    
    if test_file is not None:
        # Load image
        image = Image.open(test_file)
        
        # Detect coins
        with st.spinner("Detecting and classifying coins..."):
            coin_regions, annotated_img = detect_coins_in_image(image)
            
            if coin_regions:
                # Classify each detected coin
                results = []
                for coin in coin_regions:
                    coin_type, confidence = classify_coin(
                        coin['roi'],
                        st.session_state.classifier,
                        st.session_state.scaler
                    )
                    results.append({
                        'type': coin_type,
                        'confidence': confidence,
                        'position': coin['position']
                    })
                
                # Update totals
                old_in_image = sum(1 for r in results if r['type'] == 'OLD')
                new_in_image = sum(1 for r in results if r['type'] == 'NEW')
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Detection Result")
                    st.image(annotated_img, use_container_width=True)
                    st.caption("Green circles show detected coin positions")
                
                with col2:
                    st.subheader("Detection Summary")
                    st.write(f"**Coins detected:** {len(results)}")
                    st.write(f"**OLD 1 Peso coins:** {old_in_image}")
                    st.write(f"**NEW 1 Peso coins:** {new_in_image}")
                    st.write(f"**Total value:** PHP {old_in_image + new_in_image}.00")
                    
                    st.subheader("Detailed Results")
                    for i, res in enumerate(results, 1):
                        confidence_pct = res['confidence'] * 100
                        st.write(f"Coin {i}: **{res['type']}** (confidence: {confidence_pct:.1f}%)")
                
                # Add to running totals
                st.markdown("---")
                st.subheader("Add to Running Total")
                
                col_add1, col_add2, col_add3 = st.columns(3)
                with col_add1:
                    if st.button(f"Add {old_in_image} OLD Coin(s)", use_container_width=True):
                        st.session_state.total_old += old_in_image
                        st.session_state.total_new += new_in_image
                        st.session_state.total_value = st.session_state.total_old + st.session_state.total_new
                        st.success(f"Added {old_in_image} OLD and {new_in_image} NEW coin(s)")
                        st.rerun()
                
                with col_add2:
                    if st.button("Discard Results", use_container_width=True):
                        st.rerun()
                
                with col_add3:
                    if st.button("Reset All Counters", use_container_width=True):
                        st.session_state.total_old = 0
                        st.session_state.total_new = 0
                        st.session_state.total_value = 0
                        st.success("All counters reset")
                        st.rerun()
                        
            else:
                st.warning("No coins detected in the image.")
                st.info("Tips: Use clear, well-lit images with coins on a plain background. Make sure coins are not overlapping.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>PisoUno</strong> - Philippine 1 Peso Coin Detection System</p>
    <p>Features: Denomination Classification | Old vs New | Multiple Coin Detection | Counting | Total Value</p>
</div>
""", unsafe_allow_html=True)
