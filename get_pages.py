# get_pages.py
def get_all_pages():
    return [
        "Home",
        "Image Detection", 
        "Video Detection",
        "Multiple Images",
        "Webcam Detection",
        "Eco Mode",
        "DeepSort"
    ]

# Page configurations with emojis and descriptions
def get_page_config():
    return {
        "Home": {
            "icon": "🏠",
            "description": "Main dashboard and overview",
            "category": "main"
        },
        "Image Detection": {
            "icon": "📸",
            "description": "Single image object detection",
            "category": "detection"
        },
        "Video Detection": {
            "icon": "🎥",
            "description": "Video file processing and analysis",
            "category": "detection"
        },
        "Multiple Images": {
            "icon": "🖼️",
            "description": "Batch image processing",
            "category": "detection"
        },
        "Webcam Detection": {
            "icon": "📹",
            "description": "Real-time webcam detection",
            "category": "detection"
        },
        "Eco Mode": {
            "icon": "🌱",
            "description": "Energy-efficient detection mode",
            "category": "optimization"
        },
        "DeepSort": {
            "icon": "🎯",
            "description": "Advanced object tracking with DeepSort",
            "category": "tracking"
        }
    }
# get_pages.py

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class PageCategory(Enum):
    """Enumeration for page categories"""
    MAIN = "main"
    DETECTION = "detection"
    PROCESSING = "processing"
    OPTIMIZATION = "optimization"
    TRACKING = "tracking"
    ANALYSIS = "analysis"
    UTILITIES = "utilities"

class ProcessingMode(Enum):
    """Enumeration for processing modes"""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAM = "stream"
    INTERACTIVE = "interactive"

@dataclass
class PageMetadata:
    """Extended metadata for pages"""
    icon: str
    description: str
    category: PageCategory
    processing_mode: ProcessingMode
    is_gpu_intensive: bool = False
    requires_camera: bool = False
    supports_batch: bool = False
    min_memory_gb: float = 1.0
    estimated_processing_time: str = "< 1s"
    supported_formats: List[str] = None
    features: List[str] = None

def get_all_pages() -> List[str]:
    """
    Returns a list of all available pages in the application.
    
    Returns:
        List[str]: List of page names
    """
    return [
        "Home",
        "Image Detection", 
        "Video Detection",
        "Multiple Images",
        "Webcam Detection",
        "Eco Mode",
        "DeepSort"
    ]

def get_page_config() -> Dict[str, Dict[str, Any]]:
    """
    Returns comprehensive page configuration with enhanced metadata.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping page names to their configurations
    """
    return {
        "Home": {
            "icon": "🏠",
            "description": "Main dashboard with system overview, performance metrics, and navigation center",
            "category": "main",
            "processing_mode": "interactive",
            "is_gpu_intensive": False,
            "requires_camera": False,
            "supports_batch": False,
            "min_memory_gb": 0.5,
            "estimated_processing_time": "instant",
            "supported_formats": [],
            "features": [
                "System monitoring",
                "Performance metrics", 
                "Resource utilization",
                "Quick navigation",
                "Status dashboard"
            ],
            "keywords": ["dashboard", "overview", "metrics", "system", "home"],
            "priority": 1
        },
        
        "Image Detection": {
            "icon": "📸",
            "description": "Advanced single image analysis with AI-powered object detection and classification",
            "category": "detection", 
            "processing_mode": "interactive",
            "is_gpu_intensive": True,
            "requires_camera": False,
            "supports_batch": False,
            "min_memory_gb": 2.0,
            "estimated_processing_time": "0.5-2s",
            "supported_formats": [
                "JPG", "JPEG", "PNG", "BMP", "TIFF", "WEBP"
            ],
            "features": [
                "Single image upload",
                "Real-time object detection",
                "Confidence scoring",
                "Bounding box visualization", 
                "Class predictions",
                "Downloadable results"
            ],
            "keywords": ["image", "photo", "detection", "upload", "single"],
            "priority": 2
        },
        
        "Video Detection": {
            "icon": "🎥",
            "description": "Comprehensive video file processing with frame-by-frame analysis and timeline controls",
            "category": "detection",
            "processing_mode": "batch",
            "is_gpu_intensive": True,
            "requires_camera": False,
            "supports_batch": True,
            "min_memory_gb": 4.0,
            "estimated_processing_time": "30s-5m",
            "supported_formats": [
                "MP4", "AVI", "MOV", "WMV", "FLV", "MKV", "WEBM"
            ],
            "features": [
                "Video file upload",
                "Frame-by-frame analysis",
                "Timeline scrubbing",
                "Object tracking",
                "Export processed video",
                "Performance analytics"
            ],
            "keywords": ["video", "file", "processing", "timeline", "frames"],
            "priority": 3
        },
        
        "Multiple Images": {
            "icon": "🖼️",
            "description": "Efficient batch processing for multiple images with parallel processing capabilities",
            "category": "processing",
            "processing_mode": "batch",
            "is_gpu_intensive": True,
            "requires_camera": False,
            "supports_batch": True,
            "min_memory_gb": 3.0,
            "estimated_processing_time": "1-10s per image",
            "supported_formats": [
                "JPG", "JPEG", "PNG", "BMP", "TIFF", "WEBP"
            ],
            "features": [
                "Multiple image upload",
                "Batch processing",
                "Parallel execution",
                "Progress tracking",
                "Bulk export",
                "Statistical summaries"
            ],
            "keywords": ["batch", "multiple", "bulk", "parallel", "images"],
            "priority": 4
        },
        
        "Webcam Detection": {
            "icon": "📹",
            "description": "Real-time camera feed analysis with live object detection and tracking",
            "category": "detection",
            "processing_mode": "realtime",
            "is_gpu_intensive": True,
            "requires_camera": True,
            "supports_batch": False,
            "min_memory_gb": 2.5,
            "estimated_processing_time": "real-time (24-30 FPS)",
            "supported_formats": [
                "Camera Feed", "USB Webcam", "IP Camera"
            ],
            "features": [
                "Live camera feed",
                "Real-time detection",
                "FPS monitoring",
                "Recording capability",
                "Snapshot capture",
                "Stream settings"
            ],
            "keywords": ["webcam", "camera", "live", "realtime", "stream"],
            "priority": 5,
            "requirements": {
                "camera_access": True,
                "min_fps": 15,
                "recommended_resolution": "720p"
            }
        },
        
        "Eco Mode": {
            "icon": "🌱",
            "description": "Energy-efficient detection mode with adaptive processing and power optimization",
            "category": "optimization",
            "processing_mode": "interactive",
            "is_gpu_intensive": False,
            "requires_camera": False,
            "supports_batch": True,
            "min_memory_gb": 1.5,
            "estimated_processing_time": "2-5s (optimized)",
            "supported_formats": [
                "JPG", "JPEG", "PNG", "MP4", "AVI"
            ],
            "features": [
                "Power optimization",
                "Adaptive processing",
                "Resource monitoring",
                "Battery-friendly mode",
                "Quality vs speed balance",
                "Green computing metrics"
            ],
            "keywords": ["eco", "green", "efficient", "battery", "optimization"],
            "priority": 6,
            "benefits": {
                "power_savings": "30-50%",
                "heat_reduction": "25%",
                "battery_life_extension": "2x"
            }
        },
        
        "DeepSort": {
            "icon": "🎯",
            "description": "Advanced multi-object tracking with persistent IDs and trajectory analysis",
            "category": "tracking", 
            "processing_mode": "stream",
            "is_gpu_intensive": True,
            "requires_camera": False,
            "supports_batch": True,
            "min_memory_gb": 3.5,
            "estimated_processing_time": "real-time tracking",
            "supported_formats": [
                "MP4", "AVI", "MOV", "Camera Feed", "RTSP Stream"
            ],
            "features": [
                "Multi-object tracking",
                "Persistent ID assignment",
                "Trajectory analysis",
                "Re-identification",
                "Track visualization",
                "Advanced analytics"
            ],
            "keywords": ["tracking", "deepsort", "trajectory", "analytics", "ids"],
            "priority": 7,
            "advanced_features": {
                "kalman_filtering": True,
                "appearance_modeling": True,
                "occlusion_handling": True,
                "track_prediction": True
            }
        }
    }

def get_pages_by_category(category: PageCategory) -> List[str]:
    """
    Get all pages belonging to a specific category.
    
    Args:
        category (PageCategory): The category to filter by
        
    Returns:
        List[str]: List of page names in the specified category
    """
    config = get_page_config()
    return [
        page_name for page_name, page_data in config.items() 
        if page_data.get("category") == category.value
    ]

def get_gpu_intensive_pages() -> List[str]:
    """
    Get all pages that require GPU processing.
    
    Returns:
        List[str]: List of GPU-intensive page names
    """
    config = get_page_config()
    return [
        page_name for page_name, page_data in config.items()
        if page_data.get("is_gpu_intensive", False)
    ]

def get_camera_required_pages() -> List[str]:
    """
    Get all pages that require camera access.
    
    Returns:
        List[str]: List of page names requiring camera
    """
    config = get_page_config()
    return [
        page_name for page_name, page_data in config.items()
        if page_data.get("requires_camera", False)
    ]

def get_batch_processing_pages() -> List[str]:
    """
    Get all pages that support batch processing.
    
    Returns:
        List[str]: List of page names supporting batch processing
    """
    config = get_page_config()
    return [
        page_name for page_name, page_data in config.items()
        if page_data.get("supports_batch", False)
    ]

def get_page_requirements(page_name: str) -> Dict[str, Any]:
    """
    Get system requirements for a specific page.
    
    Args:
        page_name (str): Name of the page
        
    Returns:
        Dict[str, Any]: System requirements and recommendations
    """
    config = get_page_config()
    page_data = config.get(page_name, {})
    
    return {
        "min_memory_gb": page_data.get("min_memory_gb", 1.0),
        "is_gpu_intensive": page_data.get("is_gpu_intensive", False),
        "requires_camera": page_data.get("requires_camera", False),
        "estimated_processing_time": page_data.get("estimated_processing_time", "unknown"),
        "supported_formats": page_data.get("supported_formats", []),
        "additional_requirements": page_data.get("requirements", {})
    }

def search_pages(query: str) -> List[tuple]:
    """
    Search pages by keywords and descriptions.
    
    Args:
        query (str): Search query
        
    Returns:
        List[tuple]: List of (page_name, relevance_score) tuples
    """
    config = get_page_config()
    query_lower = query.lower()
    results = []
    
    for page_name, page_data in config.items():
        score = 0
        
        # Check in keywords
        keywords = page_data.get("keywords", [])
        for keyword in keywords:
            if query_lower in keyword.lower():
                score += 10
                
        # Check in description
        description = page_data.get("description", "").lower()
        if query_lower in description:
            score += 5
            
        # Check in features
        features = page_data.get("features", [])
        for feature in features:
            if query_lower in feature.lower():
                score += 3
                
        # Check in page name
        if query_lower in page_name.lower():
            score += 15
            
        if score > 0:
            results.append((page_name, score))
    
    # Sort by relevance score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_navigation_menu() -> Dict[str, List[str]]:
    """
    Get organized navigation menu structure.
    
    Returns:
        Dict[str, List[str]]: Navigation menu organized by categories
    """
    config = get_page_config()
    menu = {}
    
    # Group pages by category
    for page_name, page_data in config.items():
        category = page_data.get("category", "main")
        category_title = category.replace("_", " ").title()
        
        if category_title not in menu:
            menu[category_title] = []
        
        # Add page with icon and priority for sorting
        priority = page_data.get("priority", 999)
        menu[category_title].append((page_name, page_data.get("icon", "📄"), priority))
    
    # Sort pages within each category by priority
    for category in menu:
        menu[category].sort(key=lambda x: x[2])  # Sort by priority
        menu[category] = [(name, icon) for name, icon, _ in menu[category]]  # Remove priority from final result
    
    return menu

def get_page_statistics() -> Dict[str, Any]:
    """
    Get statistics about all pages.
    
    Returns:
        Dict[str, Any]: Various statistics about the pages
    """
    config = get_page_config()
    
    total_pages = len(config)
    gpu_intensive = len(get_gpu_intensive_pages())
    camera_required = len(get_camera_required_pages())
    batch_support = len(get_batch_processing_pages())
    
    categories = {}
    processing_modes = {}
    
    for page_data in config.values():
        # Count categories
        category = page_data.get("category", "unknown")
        categories[category] = categories.get(category, 0) + 1
        
        # Count processing modes
        mode = page_data.get("processing_mode", "unknown")
        processing_modes[mode] = processing_modes.get(mode, 0) + 1
    
    return {
        "total_pages": total_pages,
        "gpu_intensive_pages": gpu_intensive,
        "camera_required_pages": camera_required,
        "batch_processing_pages": batch_support,
        "categories": categories,
        "processing_modes": processing_modes,
        "coverage": {
            "detection": len(get_pages_by_category(PageCategory.DETECTION)),
            "processing": len(get_pages_by_category(PageCategory.PROCESSING)),
            "optimization": len(get_pages_by_category(PageCategory.OPTIMIZATION)),
            "tracking": len(get_pages_by_category(PageCategory.TRACKING))
        }
    }

# Example usage and testing functions
if __name__ == "__main__":
    # Test the functions
    print("=== OrbitalVision Pages Configuration ===")
    print(f"Total Pages: {len(get_all_pages())}")
    print(f"GPU Intensive: {get_gpu_intensive_pages()}")
    print(f"Camera Required: {get_camera_required_pages()}")
    print(f"Batch Processing: {get_batch_processing_pages()}")
    
    print("\n=== Search Test ===")
    search_results = search_pages("detection")
    for page, score in search_results[:3]:
        print(f"{page}: {score}")
    
    print("\n=== Navigation Menu ===")
    menu = get_navigation_menu()
    for category, pages in menu.items():
        print(f"{category}: {[f'{icon} {name}' for name, icon in pages]}")
    
    print("\n=== Statistics ===")
    stats = get_page_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")