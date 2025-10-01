# Africa Crop Doctor AI - Full Stack Web Application

A comprehensive web application showcasing the Africa Crop Doctor AI model for crop disease detection. Built with a modern tech stack including React, Vite, TailwindCSS for the frontend and Node.js/Express for the backend API.

## 🌟 Features

### Frontend (React + Vite + TailwindCSS)
- **Responsive Design** - Works seamlessly on desktop and mobile devices
- **Interactive Testing** - Upload crop images for real-time disease detection
- **Model Information** - Detailed technical specifications and architecture details
- **Training Insights** - Comprehensive training metrics and progress visualization
- **Image Gallery** - Sample images from the training dataset
- **Modern UI** - Clean, professional interface with smooth animations

### Backend (Node.js + Express)
- **AI Model API** - REST API for crop disease prediction
- **Image Processing** - Automatic image preprocessing and optimization
- **Rate Limiting** - Protection against API abuse
- **Error Handling** - Comprehensive error management and logging
- **CORS Support** - Secure cross-origin resource sharing
- **File Validation** - Secure file upload with type and size validation

### AI Model Capabilities
- **85.6% Disease Accuracy** - High-precision disease classification
- **35 Disease Classes** - Comprehensive disease detection across major crops
- **13 Crop Types** - Support for African and international crops
- **Treatment Recommendations** - Actionable advice for disease management
- **Multiple Output Formats** - Chemical, organic, and cultural treatments

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ installed
- npm or yarn package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd africa-crop-doctor-website
```

2. **Install Frontend Dependencies**
```bash
cd frontend
npm install
```

3. **Install Backend Dependencies**
```bash
cd ../backend
npm install
```

4. **Environment Setup**
```bash
# In backend directory
cp .env.example .env
# Edit .env with your configuration if needed
```

5. **Start Development Servers**

**Terminal 1 - Backend:**
```bash
cd backend
npm run dev
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

6. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## 🏗️ Project Structure

```
africa-crop-doctor-website/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   ├── components/         # Reusable React components
│   │   │   ├── Header.jsx      # Navigation header
│   │   │   └── Footer.jsx      # Site footer
│   │   ├── pages/              # Main application pages
│   │   │   ├── Home.jsx        # Landing page
│   │   │   ├── ModelInfo.jsx   # Model specifications
│   │   │   ├── Gallery.jsx     # Dataset gallery
│   │   │   ├── Training.jsx    # Training metrics
│   │   │   └── TestModel.jsx   # Interactive testing
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # Application entry point
│   ├── package.json            # Frontend dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── tailwind.config.js      # TailwindCSS configuration
│   └── index.html              # HTML template
├── backend/                    # Node.js + Express backend
│   ├── server.js               # Main server file
│   ├── package.json            # Backend dependencies
│   ├── .env.example            # Environment template
│   └── README.md               # Backend documentation
└── README.md                   # This file
```

## 🔧 Configuration

### Frontend Configuration
- **Vite Config**: `frontend/vite.config.js`
- **TailwindCSS**: `frontend/tailwind.config.js`
- **PostCSS**: `frontend/postcss.config.js`

### Backend Configuration
- **Environment Variables**: `backend/.env`
- **CORS Origins**: Configure allowed origins in `.env`
- **Rate Limits**: Adjust in `server.js`

## 📡 API Endpoints

### Model Information
- `GET /api/health` - Health check
- `GET /api/model/info` - Model specifications
- `GET /api/classes` - Supported crops and diseases

### Prediction
- `POST /api/predict` - Upload image for disease detection
  - **Input**: Form data with `image` field
  - **Formats**: JPG, PNG, WebP (max 10MB)
  - **Output**: Crop type, disease, confidence, treatments

### Example API Usage
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

## 🎨 UI Components

### Pages Overview
1. **Home** - Hero section, statistics, features showcase
2. **Model Info** - Architecture details, performance metrics, technical specs
3. **Gallery** - Dataset samples with filtering and search
4. **Training** - Training progress, hyperparameters, dataset information
5. **Test Model** - Interactive image upload and prediction interface

### Design System
- **Colors**: Green primary theme representing agriculture
- **Typography**: Clean, readable fonts optimized for technical content
- **Icons**: Lucide React icons for consistency
- **Animations**: Smooth transitions and hover effects
- **Responsive**: Mobile-first design approach

## 🔒 Security Features

- **Helmet.js** - Security headers
- **Rate Limiting** - API abuse protection
- **File Validation** - Secure uploads
- **CORS Protection** - Cross-origin security
- **Input Sanitization** - XSS prevention

## 🧪 Testing the Application

### Manual Testing
1. Start both frontend and backend servers
2. Navigate to http://localhost:5173
3. Explore different pages using the navigation
4. Test image upload on the "Test Model" page
5. Verify API responses in browser developer tools

### API Testing
```bash
# Health check
curl http://localhost:5000/api/health

# Model info
curl http://localhost:5000/api/model/info

# Prediction (replace with actual image path)
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/api/predict
```

## 🚀 Deployment

### Frontend Deployment (Netlify/Vercel)
```bash
cd frontend
npm run build
# Deploy the 'dist' folder
```

### Backend Deployment (Railway/Heroku)
```bash
cd backend
# Set production environment variables
npm install --production
npm start
```

### Environment Variables for Production
```bash
NODE_ENV=production
PORT=5000
CORS_ORIGINS=https://your-frontend-domain.com
```

## ⚡ Performance Optimizations

### Frontend
- **Vite Build Optimization** - Fast builds and hot module replacement
- **Code Splitting** - Lazy loading for optimal bundle size
- **Image Optimization** - Compressed images and proper formats
- **CSS Purging** - TailwindCSS removes unused styles

### Backend
- **Image Processing** - Sharp.js for efficient image handling
- **Compression** - Gzip compression for API responses
- **Caching Headers** - Proper HTTP caching
- **Request Validation** - Early rejection of invalid requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and structure
- Add proper error handling for new features
- Update documentation for API changes
- Test thoroughly before submitting

## 📊 Model Integration

The current implementation uses mock predictions for demonstration. To integrate with your actual trained model:

1. **Install ML Runtime** (PyTorch Node.js, ONNX Runtime, or TensorFlow.js)
2. **Load Model** in `backend/server.js`
3. **Update Prediction Function** with real inference
4. **Add Preprocessing** to match training requirements

Example integration:
```javascript
const ort = require('onnxruntime-node');

async function loadModel() {
  const session = await ort.InferenceSession.create('./models/model.onnx');
  return session;
}

async function predictCropDisease(imagePath, session) {
  // Preprocess image
  // Run inference
  // Return predictions
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset Sources**: PlantDoc, PlantDisease, Custom African Collection
- **UI Inspiration**: Modern agricultural tech applications
- **Icons**: Lucide React icon library
- **Styling**: TailwindCSS utility framework

## 📧 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in each component

---

**Africa Crop Doctor AI** - Empowering African agriculture with AI-powered crop disease detection.