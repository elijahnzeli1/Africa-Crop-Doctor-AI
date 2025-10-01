# Africa Crop Doctor AI - Backend

This is the backend API server for the Africa Crop Doctor AI application, providing crop disease detection and prediction services.

## Features

- 🤖 AI-powered crop disease prediction
- 📸 Image upload and processing
- 🔒 Rate limiting and security
- 📊 Model information endpoints
- 🌍 CORS enabled for frontend integration
- 📝 Comprehensive error handling

## API Endpoints

### Health & Info
- `GET /` - API information
- `GET /api/health` - Health check
- `GET /api/model/info` - Model details and specifications

### Prediction
- `POST /api/predict` - Upload image for crop disease prediction
  - Form data with `image` field
  - Supports JPG, PNG, WebP (max 10MB)
  - Returns crop type, disease, confidence, and treatment recommendations

### Classes
- `GET /api/classes` - Get supported crop and disease classes

## Setup Instructions

### Prerequisites
- Node.js 16+ installed
- npm or yarn package manager

### Installation

1. **Navigate to backend directory**
```bash
cd africa-crop-doctor-website/backend
```

2. **Install dependencies**
```bash
npm install
```

3. **Environment setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Default settings work for development
```

4. **Start development server**
```bash
npm run dev
```

The server will start on `http://localhost:5000`

### Production Deployment

```bash
# Install production dependencies
npm install --production

# Start production server
npm start
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `development` |
| `PORT` | Server port | `5000` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173` |
| `MAX_FILE_SIZE` | Max upload size in bytes | `10485760` (10MB) |

## API Usage Examples

### Upload Image for Prediction

```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.prediction);
```

### Response Format

```json
{
  "prediction": {
    "crop": "Tomato",
    "disease": "Early blight",
    "confidence": 87.5,
    "status": "diseased",
    "severity": "Moderate"
  },
  "recommendations": [
    "Remove affected leaves immediately",
    "Apply copper-based fungicide every 7-14 days",
    "Improve air circulation around plants"
  ],
  "treatment": {
    "chemical": "Copper hydroxide or Mancozeb",
    "organic": "Baking soda spray",
    "cultural": "Crop rotation, proper spacing"
  },
  "metadata": {
    "model_version": "1.0.0",
    "processing_time": 1500,
    "timestamp": "2025-01-11T10:30:00.000Z"
  }
}
```

## Rate Limits

- General API: 100 requests per 15 minutes per IP
- Prediction endpoint: 10 requests per 5 minutes per IP

## Error Handling

The API returns structured error responses:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

Common error codes:
- `400` - Bad request (invalid file, missing parameters)
- `413` - File too large
- `429` - Rate limit exceeded
- `500` - Internal server error

## Security Features

- Helmet.js for security headers
- CORS protection
- Rate limiting
- File type validation
- File size limits
- Input sanitization

## Model Integration

The current implementation includes a mock prediction function. To integrate with your actual trained model:

1. Install PyTorch/ONNX runtime for Node.js
2. Load your model in the `predictCropDisease` function
3. Implement proper image preprocessing
4. Update the prediction logic

## Development

### Scripts
- `npm run dev` - Start development server with auto-reload
- `npm start` - Start production server
- `npm test` - Run tests (when implemented)

### File Structure
```
backend/
├── server.js          # Main server file
├── package.json       # Dependencies and scripts
├── .env.example       # Environment template
├── uploads/           # Temporary image storage
└── README.md          # This file
```

## Contributing

1. Follow existing code style
2. Add proper error handling
3. Update documentation for new endpoints
4. Test thoroughly before submitting

## License

MIT License - See LICENSE file for details