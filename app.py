# app.py - PisoUno Coin Detection System
# Optimized for Streamlit Cloud deployment

import streamlit as st
import os
import sys

# Display Python version for debugging
st.write(f"Python version: {sys.version[:5]}")

# Try importing with fallbacks
try:
    import cv2
    st.success("OpenCV loaded successfully")
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV not loaded: {e}")
    st.warning("The app will run in limited mode without OpenCV")
    CV2_AVAILABLE = False

try:
    import numpy as np
    st.success("NumPy loaded successfully")
    NUMPY_AVAILABLE = True
except ImportError as e:
    st.error(f"NumPy not loaded: {e}")
    NUMPY_AVAILABLE = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    st.success("Scikit-learn loaded successfully")
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn not loaded: {e}")
    SKLEARN_AVAILABLE = False

try:
    from PIL import Image
    st.success("PIL loaded successfully")
    PIL_AVAILABLE = True
except ImportError as e:
    st.error(f"PIL not loaded: {e}")
    PIL_AVAILABLE = False

# Check if all required packages are available
if not (CV2_AVAILABLE and NUMPY_AVAILABLE and SKLEARN_AVAILABLE and PIL_AVAILABLE):
    st.error("Missing required packages. Please check deployment logs.")
    st.stop()

# ============================================
# Feature Extraction Functions
# ============================================

def extract_coin_features(image, img_size=(64, 64)):
    """
    Extract features from a coin image for classification
    """
    try:
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_resized = cv2.resize(image, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Hu Moments (shape features)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 2. Color Histograms
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
        
        # Combine features
        features = np.hstack([hu_moments, hist_features, [edge_density, mean_intensity, std_intensity]])
        
        # Fixed size
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
                    
                    cv2.circle(annotated_img, (x, y), r, (0, 255, 0), 2)
        
        return coin_regions, annotated_img
        
    except Exception as e:
        return [], None

def train_classifier_from_uploads(old_images, new_images):
    """Train SVM classifier from uploaded images"""
    X = []
    y = []
    
    for img in old_images:
        features = extract_coin_features(img)
        if features is not None:
            X.append(features)
            y.append(0)
    
    for img in new_images:
        features = extract_coin_features(img)
        if features is not None:
            X.append(features)
            y.append(1)
    
    if len(X) < 4:
        return None, None, 0
    
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifier = SVC(kernel='rbf', probability=True, random_state=42)
    classifier.fit(X_scaled, y)
    
    accuracy = classifier.score(X_scaled, y)
    
    return classifier, scaler, accuracy

def classify_coin(coin_roi, classifier, scaler):
    """Classify a single coin as OLD or NEW"""
    try:
        features = extract_coin_features(coin_roi)
        if features is None:
            return "UNKNOWN", 0.0
        
        features_scaled = scaler.transform([features])
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return "NEW" if prediction == 1 else "OLD", confidence
        
    except Exception as e:
        return "UNKNOWN", 0.0

# ============================================
# Streamlit UI
# ============================================

st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>PisoUno</h1>
    <p>Old and New 1 Peso Coin Detection, Classification, and Counting System</p>
    <p>Value per coin: PHP 1.00</p>
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

# Sidebar
with st.sidebar:
    st.header("Training")
    
    old_train = st.file_uploader(
        "OLD 1 Peso Coins",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="old"
    )
    
    new_train = st.file_uploader(
        "NEW 1 Peso Coins",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="new"
    )
    
    if st.button("Train Classifier", type="primary"):
        if old_train and new_train and len(old_train) >= 2 and len(new_train) >= 2:
            with st.spinner("Training..."):
                old_imgs = [Image.open(f) for f in old_train]
                new_imgs = [Image.open(f) for f in new_train]
                
                classifier, scaler, acc = train_classifier_from_uploads(old_imgs, new_imgs)
                
                if classifier:
                    st.session_state.classifier = classifier
                    st.session_state.scaler = scaler
                    st.session_state.trained = True
                    st.success(f"Training complete! Accuracy: {acc:.1%}")
                else:
                    st.error("Training failed")
        else:
            st.warning("Need at least 2 images per type")
    
    if st.session_state.trained:
        st.divider()
        if st.button("Reset All Counters"):
            st.session_state.total_old = 0
            st.session_state.total_new = 0
            st.session_state.total_value = 0
            st.rerun()

# Main area
if not st.session_state.trained:
    st.warning("Train the classifier first using the sidebar")
else:
    # Display totals
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="coin-card"><div>OLD COINS</div><div class="old-coin">{st.session_state.total_old}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="coin-card"><div>NEW COINS</div><div class="new-coin">{st.session_state.total_new}</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="coin-card"><div>TOTAL VALUE</div><div class="total-value">PHP {st.session_state.total_value}.00</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Test image
    test_file = st.file_uploader("Upload coin image for detection", type=['jpg', 'jpeg', 'png'])
    
    if test_file:
        image = Image.open(test_file)
        
        with st.spinner("Detecting coins..."):
            coin_regions, annotated = detect_coins_in_image(image)
            
            if coin_regions:
                results = []
                for coin in coin_regions:
                    coin_type, conf = classify_coin(coin['roi'], st.session_state.classifier, st.session_state.scaler)
                    results.append({'type': coin_type, 'confidence': conf})
                
                old_in_img = sum(1 for r in results if r['type'] == 'OLD')
                new_in_img = sum(1 for r in results if r['type'] == 'NEW')
                
                col_img, col_res = st.columns([2, 1])
                
                with col_img:
                    st.image(annotated, use_container_width=True)
                    st.caption(f"Detected {len(results)} coin(s)")
                
                with col_res:
                    st.write(f"**OLD coins:** {old_in_img}")
                    st.write(f"**NEW coins:** {new_in_img}")
                    st.write(f"**Value:** PHP {old_in_img + new_in_img}.00")
                    
                    if st.button("Add to Total"):
                        st.session_state.total_old += old_in_img
                        st.session_state.total_new += new_in_img
                        st.session_state.total_value = st.session_state.total_old + st.session_state.total_new
                        st.success("Added! Refresh to see updated totals.")
                        st.rerun()
            else:
                st.warning("No coins detected")

st.markdown("---")
st.caption("PisoUno - Philippine 1 Peso Coin Detection System")
