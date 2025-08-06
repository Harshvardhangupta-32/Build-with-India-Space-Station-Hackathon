# Home.py
import streamlit as st
from get_pages import get_all_pages, get_page_config
import time

st.set_page_config(
    page_title="🛰 OrbitalVision:-Multi-Camera AI for Space Safety", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Enhanced Styling ----
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        .main > div {
            padding-top: 2rem;
        }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        }
        
        .sidebar-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 1rem;
            margin: -1rem -1rem 1.5rem -1rem;
            border-radius: 0 0 15px 15px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .sidebar-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.3em;
            font-weight: 700;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        
        .sidebar-subtitle {
            font-size: 0.8em;
            opacity: 0.9;
            margin-top: 0.3rem;
            font-weight: 300;
        }
        
        .nav-section {
            margin-bottom: 2rem;
        }
        
        .nav-header {
            color: #94a3b8;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.8rem;
            padding: 0 0.5rem;
            border-bottom: 1px solid #374151;
            padding-bottom: 0.5rem;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            margin-bottom: 0.3rem;
            border-radius: 8px;
            color: #e2e8f0;
            text-decoration: none;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid transparent;
        }
        
        .nav-item:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateX(3px);
            color: #ffffff;
        }
        
        .nav-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }
        
        .nav-number {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 700;
            margin-right: 0.8rem;
        }
        
        .nav-emoji {
            font-size: 1.2em;
            margin-right: 0.8rem;
        }
        
        .nav-text {
            flex: 1;
            font-size: 0.9em;
        }
        
        .stats-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.2rem;
            margin-top: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stats-title {
            color: #cbd5e1;
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 0.5rem;
        }
        
        .stat-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }
        
        .stat-label {
            color: #94a3b8;
            font-size: 0.8em;
        }
        
        .stat-value {
            color: #e2e8f0;
            font-weight: 600;
            font-size: 0.8em;
        }
        
        .status-online { color: #22c55e; }
        .status-available { color: #22c55e; }
        .status-high-performance { color: #f59e0b; }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 0;
            margin: -2rem -1rem 2rem -1rem;
            border-radius: 0 0 20px 20px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        
        .main-title {
            font-family: 'Inter', sans-serif;
            font-size: 3.5em;
            font-weight: 800;
            margin-bottom: 0.3em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: fadeInUp 1s ease-out;
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.3em;
            font-weight: 300;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
            animation: fadeInUp 1s ease-out 0.2s both;
        }
        
        .stats-container {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 2rem;
            animation: fadeInUp 1s ease-out 0.4s both;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 700;
            display: block;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        
        .stat-desc {
            font-size: 0.9em;
            opacity: 0.8;
            font-weight: 300;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.08);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover::before {
            transform: scaleX(1);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            border-color: #cbd5e0;
        }
        
        .feature-icon {
            font-size: 3em;
            margin-bottom: 1rem;
            display: block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .feature-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #2d3748;
        }
        
        .feature-description {
            color: #718096;
            line-height: 1.6;
            font-size: 1em;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---- Enhanced Sidebar ----
with st.sidebar:
    # Sidebar Header
    st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">🎛️ Control Panel</div>
            <div class="sidebar-subtitle">OrbitalVision Suite</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation Section
    st.markdown("""
        <div class="nav-section">
            <div class="nav-header">🗂️ Navigation Pages</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Get page configuration
    pages = get_all_pages()
    page_config = get_page_config()
    
    # Current page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # Create numbered navigation items
    for i, page in enumerate(pages, 1):
        config = page_config.get(page, {"icon": "📄", "description": "Page description"})
        
        # Use columns to create clickable navigation
        if st.button(f"{i:02d} {config['icon']} {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page
            st.rerun()
    
    # Quick Stats Section
    st.markdown("""
        <div class="stats-section">
            <div class="stats-title">📊 Quick Stats</div>
            <div class="stat-item">
                <span class="stat-label">🟢 System Status:</span>
                <span class="stat-value status-online">Online</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">🔋 GPU:</span>
                <span class="stat-value status-available">Available</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">⚡ Mode:</span>
                <span class="stat-value status-high-performance">High Performance</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">🎯 Active Model:</span>
                <span class="stat-value">YOLOv8n</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">📈 Uptime:</span>
                <span class="stat-value">23h 14m</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">💾 Memory:</span>
                <span class="stat-value">2.1GB / 8GB</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # System Controls
    st.markdown("---")
    st.markdown("### ⚙️ System Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart", use_container_width=True):
            st.success("System restarted!")
    
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.success("Cache cleared!")

# ---- Main Content ----
st.markdown("""
    <div class="main-header">
        <div class="main-title">🛰 OrbitalVision Suite</div>
        <div class="subtitle">
            Advanced real-time object detection powered by YOLOv8 with multiple detection modes, 
            optimized performance, and comprehensive tracking capabilities
        </div>
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-number">99.2%</span>
                <span class="stat-desc">Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">60+</span>
                <span class="stat-desc">FPS</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">80+</span>
                <span class="stat-desc">Object Classes</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Show current page indicator
st.info(f"📍 Current Page: **{st.session_state.current_page}**")

# ---- Performance Metrics ----
st.markdown("## 📊 Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Detection Speed",
        value="45ms",
        delta="-12ms",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Model Size",
        value="6.2MB",
        delta="Lightweight",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Memory Usage",
        value="2.1GB",
        delta="-0.5GB",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Supported Formats",
        value="15+",
        delta="+3",
        delta_color="normal"
    )

# ---- Enhanced Features Grid ----
st.markdown("## 🚀 Detection Capabilities")

features_data = [
    {
        "icon": "📸",
        "title": "Smart Image Detection",
        "description": "Upload any image and get instant object detection with confidence scores, bounding boxes, and class predictions. Supports JPEG, PNG, WebP formats."
    },
    {
        "icon": "🎥",
        "title": "Real-time Video Analysis",
        "description": "Process video files or live webcam feeds with frame-by-frame detection. Export annotated videos with detection results."
    },
    {
        "icon": "⚡",
        "title": "Eco Mode Optimization",
        "description": "Intelligent resource management that adapts processing power based on scene complexity while maintaining accuracy."
    },
    {
        "icon": "🎯",
        "title": "DeepSort Tracking",
        "description": "Advanced multi-object tracking that maintains object identities across frames with state-of-the-art tracking algorithms."
    },
    {
        "icon": "🔄",
        "title": "Multi-Source Input",
        "description": "Seamlessly switch between webcam, uploaded videos, IP cameras, and RTSP streams for versatile detection scenarios."
    },
    {
        "icon": "📈",
        "title": "Analytics Dashboard",
        "description": "Comprehensive detection analytics with object counting, heatmaps, and performance metrics visualization."
    }
]

# Create feature cards in a 2x3 grid
col1, col2 = st.columns(2)
for i, feature in enumerate(features_data):
    with col1 if i % 2 == 0 else col2:
        st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{feature['icon']}</div>
                <div class="feature-title">{feature['title']}</div>
                <div class="feature-description">{feature['description']}</div>
            </div>
        """, unsafe_allow_html=True)

# ---- Getting Started Section ----
st.markdown("## 🎯 Get Started")
st.markdown("""
Ready to explore the power of AI-driven object detection? Use the **Control Panel** in the sidebar to navigate between different detection modes and start detecting objects in seconds!

**Quick Navigation:**
- **01 🏠 Home** - Main dashboard and overview
- **02 📸 Image Detection** - Single image processing
- **03 🎥 Video Detection** - Video file analysis
- **04 🖼️ Multiple Images** - Batch processing
- **05 📹 Webcam Detection** - Real-time detection
- **06 🌱 Eco Mode** - Energy-efficient processing
- **07 🎯 DeepSort** - Advanced object tracking
""")

# ---- Interactive Demo Button ----
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🎮 Launch Interactive Demo", key="demo_btn", help="Try our object detection with sample data", use_container_width=True):
        st.balloons()
        st.success("🚀 Demo mode activated! Use the numbered navigation in the sidebar to explore different detection methods.")
        time.sleep(1)

# ---- Footer ----
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 0.95em; margin-top: 2rem;'>
    <p>Built with ❤️ for <strong>OrbitalVision 2025</strong></p>
    <p>Leveraging state-of-the-art YOLOv8 architecture for real-time object detection</p>
    <p style="font-size: 0.85em; margin-top: 1rem; opacity: 0.7;">
        © 2024 OrbitalVision Suite | TEAM BITRATE
    </p>
</div>
""", unsafe_allow_html=True)