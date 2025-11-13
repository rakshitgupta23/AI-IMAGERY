# AI Product Imagery

**Take Home Assignment: Junior Full Stack AI Developer**

An end-to-end AI-powered solution that extracts and enhances product images from YouTube videos using Google Gemini multimodal capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Design Decisions](#design-decisions)
- [Challenges & Solutions](#challenges--solutions)
- [Time Breakdown](#time-breakdown)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This application processes YouTube product videos to automatically:
1. Extract sample frames from the video
2. Identify products using AI vision
3. Select the best frame for each product
4. Segment products with transparent backgrounds
5. Generate enhanced product images with different styles

---

## 🛠 Tech Stack

**Backend:**
- Python 3.10+
- FastAPI
- Google Gemini 2.5 Flash (multimodal AI)
- OpenCV, Pillow (image processing)
- yt-dlp/pytube (video download)

**Frontend:**
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Lucide icons

---

## ✨ Features

- **YouTube Video Processing**: Downloads and extracts frames from product videos
- **AI Product Detection**: Uses Gemini Vision to identify products in frames
- **Intelligent Frame Selection**: AI picks the best frame showing each product
- **Product Segmentation**: Segments products with transparent backgrounds
- **Image Enhancement**: Generates multiple styled product shots (studio, lifestyle, dramatic)
- **Fallback System**: Deterministic fallbacks ensure pipeline completion even without API access
- **Download Support**: Direct download of all generated images

---

## 🏗 Architecture

### Pipeline Flow

```
YouTube URL → Download Video → Extract 3 Frames (20%, 50%, 80%)
              ↓
        Identify Products (Gemini Vision)
              ↓
        Select Best Frame per Product (Gemini)
              ↓
        Segment Product (Gemini Image Generation)
              ↓
        Enhance with Styles (Gemini Image Enhancement)
              ↓
        Return Results to Frontend
```

### LangGraph-Style Pipeline

The backend implements a node-based processing pipeline:

1. **Download Node**: Fetches YouTube video using yt-dlp
2. **Extract Node**: Samples 3 frames at fixed percentiles
3. **Identify Node**: Detects products in each frame (Gemini Vision API)
4. **Select Node**: Chooses best frame per product (Gemini multi-image analysis)
5. **Segment Node**: Isolates product from background (Gemini image generation)
6. **Enhance Node**: Creates styled variations (Gemini image transformation)

Each node has deterministic fallback logic to ensure robust execution.

---

## 📦 Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Google Gemini API key (optional, works with fallbacks)
- ffmpeg (recommended for better video processing)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-dotenv pillow opencv-python-headless yt-dlp google-generativeai

# Create .env file
cp .env.example .env
# Edit .env with your settings
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd ai-imagery-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

---

## 🚀 Usage

### 1. Configure Environment Variables

Create `.env` in backend directory:

```env
# Workdir for file storage
LG_WORKDIR=./workdir

# Enable/disable Gemini (true/false)
USE_REAL_GEMINI=true

# Google API key (required if USE_REAL_GEMINI=true)
GOOGLE_API_KEY=your_api_key_here

# Pipeline limits
MAX_PRODUCTS=2
VARIATIONS=3

# Model configuration
GEMINI_VISION_MODEL=models/gemini-2.5-flash
GEMINI_IMAGE_MODEL=models/gemini-2.5-flash
```

### 2. Start Backend

```bash
cd backend
uvicorn langgraph_backend:app --reload --port 8000
```

### 3. Start Frontend

```bash
cd ai-imagery-frontend
npm run dev
```

### 4. Process a Video

1. Open http://localhost:3000
2. Paste a YouTube product video URL
3. Click "Process"
4. View and download generated images

---

## 📡 API Documentation

### POST `/api/process`

Process a YouTube video and extract product images.

**Request:**
```json
{
  "youtube_url": "https://youtube.com/shorts/xyz",
  "max_products": 2
}
```

**Response:**
```json
{
  "youtube_url": "https://youtube.com/shorts/xyz",
  "products": [
    {
      "label": "White box",
      "best_frame": "/path/to/frame.jpg",
      "cropped": "/path/to/segmented.png",
      "enhanced": [
        "/path/to/enhanced1.jpg",
        "/path/to/enhanced2.jpg",
        "/path/to/enhanced3.jpg"
      ]
    }
  ]
}
```

### GET `/files?path=<path>`

Serve generated image files.

**Parameters:**
- `path`: Absolute or relative path to image file

**Response:** Binary image file with appropriate content-type

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🎯 Design Decisions

### 1. Limited Frame Sampling
**Decision:** Extract only 3 frames at fixed positions (20%, 50%, 80%)

**Rationale:**
- Reduces API calls and processing time
- Provides good coverage of video content
- Avoids Gemini rate limits on free tier

### 2. Single-Attempt API Calls
**Decision:** Try Gemini once per operation, fallback immediately on failure

**Rationale:**
- Prevents infinite retry loops
- Ensures predictable execution time
- Makes system testable without API access
- Gemini free tier has strict rate limits

### 3. Deterministic Fallbacks
**Decision:** Implement fallback logic for every AI operation

**Rationale:**
- **Identification**: Returns generic "product" label
- **Selection**: Picks first or middle frame
- **Segmentation**: Center-crop (60% of frame)
- **Enhancement**: Resizes to 1024x1024

This ensures the pipeline always completes successfully.

### 4. Run Isolation
**Decision:** Use unique RUN_ID per request and reset output directory

**Rationale:**
- Prevents file mixing between runs
- Avoids serving stale images
- Simplifies cleanup and debugging

### 5. Product and Variation Limits
**Decision:** Default to 2 products and 3 enhancement variations

**Rationale:**
- Balances quality demonstration with API quota
- Reasonable for UI display
- Configurable via environment variables

---

## 🔧 Challenges & Solutions

### Challenge 1: Gemini Rate Limits
**Problem:** Free tier API heavily rate-limited, causing frequent 429 errors

**Solution:**
- Implemented single-attempt calls with immediate fallback
- Limited frames, products, and variations
- Added deterministic fallback for every operation
- Pipeline completes successfully even without API access

### Challenge 2: Response Parsing
**Problem:** Gemini responses have inconsistent structure (text, parts, candidates)

**Solution:**
- Created `extract_text_from_response()` to handle multiple response formats
- Try multiple extraction paths before failing
- Gracefully handle missing or malformed data

### Challenge 3: Image Data Extraction
**Problem:** Gemini image responses can be in various formats (bytes, base64, parts)

**Solution:**
- Iterate through all possible response structures
- Check for image MIME types and data fields
- Fallback to deterministic image processing on any parsing failure

### Challenge 4: File Serving Across Platforms
**Problem:** Windows absolute paths in backend, cross-platform file access

**Solution:**
- Use `pathlib.Path` for cross-platform compatibility
- Validate paths against workdir to prevent directory traversal
- Serve files with `no-store` cache headers for fresh content

### Challenge 5: Video Download Reliability
**Problem:** Different video formats, codec issues, ffmpeg warnings

**Solution:**
- Primary: yt-dlp with best MP4 format selection
- Fallback: pytube for simpler cases
- Handle missing FPS/duration gracefully in frame extraction

---

## ⏱ Time Breakdown

| Task | Time Spent |
|------|-----------|
| Backend pipeline implementation + fallbacks | 8-10 hours |
| Gemini integration & error handling | 3-4 hours |
| Frame extraction & video processing | 2 hours |
| Frontend UI development | 4-5 hours |
| Testing & debugging rate limits | 3 hours |
| Documentation & code cleanup | 2 hours |
| **Total** | **~22-26 hours** |

---

## 🚀 Future Improvements

### Performance & Scalability
- **Job Queue System**: Implement Celery/RQ for async video processing
- **Caching**: Cache results by video ID to avoid reprocessing
- **Batch Processing**: Process multiple videos simultaneously
- **Progress Tracking**: WebSocket updates for real-time progress

### AI Enhancements
- **Local Segmentation**: Integrate SAM (Segment Anything Model) for offline segmentation
- **Retry Logic**: Smart retry with exponential backoff for transient errors
- **Model Selection**: Allow users to choose between different AI models
- **Confidence Scoring**: Display AI confidence scores for product detection

### Features
- **Multiple Videos**: Process playlists or multiple URLs
- **Custom Styles**: User-defined enhancement styles
- **Export Options**: Bulk download as ZIP, different resolutions
- **History**: Save and browse previous processing jobs

### Infrastructure
- **Authentication**: User accounts and API key management
- **Cloud Storage**: S3/GCS for scalable file storage
- **CDN**: Serve images through CDN for better performance
- **Monitoring**: Logging, metrics, and error tracking
- **Docker**: Containerize for easy deployment

### Security
- **Rate Limiting**: Per-user API rate limits
- **Input Validation**: Strict URL validation and sanitization
- **Signed URLs**: Temporary authenticated file access
- **CORS**: Stricter CORS policy for production

---

## 📝 Notes

### Testing Without Gemini API
Set `USE_REAL_GEMINI=false` in `.env` to test the complete pipeline with deterministic fallbacks. This demonstrates the full workflow without requiring API access.

### API Quota Considerations
The free tier of Gemini API has strict rate limits. For production use:
- Consider paid tier with higher quotas
- Implement proper rate limiting and queuing
- Add user notifications for quota exhaustion
- Cache results aggressively

### ffmpeg Requirement
While not strictly required, installing ffmpeg improves video processing reliability and reduces codec warnings.

---

## 📄 License

This project was created as a take-home assignment for Junior Full Stack AI Developer position.

---

## 🙏 Acknowledgments

- Google Gemini API for multimodal AI capabilities
- FastAPI for excellent async API framework
- Next.js and shadcn/ui for modern frontend development
- yt-dlp for reliable video downloading