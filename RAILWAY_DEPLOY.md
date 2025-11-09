# Railway Deployment Guide

## Quick Setup

Your web demo is now ready for Railway deployment! Follow these steps:

### 1. Sign up for Railway
- Go to https://railway.app
- Sign up with your GitHub account

### 2. Create a New Project
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your repository: `moritz-rsth/AffineImageTransformer`
- Select the `web_demo` branch

### 3. Railway Auto-Detection
Railway will automatically:
- Detect Python from `requirements.txt`
- Install all dependencies
- Use the `Procfile` to start the application
- Set the `PORT` environment variable automatically

### 4. Configure Environment Variables (Optional)
Railway will automatically set:
- `PORT` - Automatically set by Railway
- `PYTHON_VERSION` - Uses Python 3.11.0 (from `runtime.txt`)

You can optionally set:
- `FLASK_DEBUG=false` - Disable debug mode (recommended for production)
- `FLASK_HOST=0.0.0.0` - Already the default in config.py

### 5. Deploy
- Railway will automatically build and deploy
- The build process installs dependencies from `requirements.txt`
- The app starts using the command in `Procfile`: `cd backend && python app.py`

### 6. Get Your URL
- Once deployed, Railway provides a public URL
- Your app will be accessible at: `https://your-app-name.up.railway.app`
- The Flask app serves both the frontend (at `/`) and API (at `/api/*`)

## What Was Configured

### Backend (`backend/config.py`)
- Uses `PORT` environment variable (Railway standard)
- Listens on `0.0.0.0` to accept external connections
- Debug mode disabled by default in production

### Frontend (`frontend/constants.js`)
- Automatically detects production environment
- Uses relative URLs when not on localhost
- No CORS issues since Flask serves both frontend and API

### Dependencies (`requirements.txt`)
- All required packages including `gunicorn` for production
- OpenCV, PyTorch, NumPy for image processing
- Flask and Flask-CORS for web framework

### Deployment Files
- `Procfile` - Tells Railway how to start the app
- `runtime.txt` - Specifies Python 3.11.0

## Troubleshooting

### Build Fails
- Check Railway logs for dependency installation errors
- Ensure all dependencies in `requirements.txt` are valid
- Large dependencies (PyTorch, OpenCV) may take time to install

### App Crashes
- Check Railway logs for runtime errors
- Verify `PORT` environment variable is set (Railway does this automatically)
- Ensure `backend/app.py` can import `image_transformation_util.py`

### API Not Working
- Verify the frontend is using relative URLs (automatically detected in production)
- Check CORS settings in `backend/app.py`
- Ensure Flask is serving the frontend correctly

## Local Testing

To test locally before deploying:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PORT=5001
export FLASK_DEBUG=false

# Run the app
cd backend && python app.py
```

Then visit: http://localhost:5001

## Notes

- Railway provides a free tier with usage limits
- Large image processing operations may take time
- The app uses in-memory image storage (images are not persisted between requests)
- For production, consider adding persistent storage for uploaded images

