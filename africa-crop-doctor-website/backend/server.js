const express = require('express');
const cors = require('cors');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const compression = require('compression');
const morgan = require('morgan');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet());
app.use(compression());
app.use(morgan('combined'));

// CORS configuration
app.use(cors({
  origin: process.env.NODE_ENV === 'production' 
    ? ['https://africa-crop-doctor.com', 'https://www.africa-crop-doctor.com']
    : ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// Stricter rate limiting for model prediction endpoint
const predictionLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, // 5 minutes
  max: 10, // limit each IP to 10 predictions per 5 minutes
  message: 'Too many prediction requests, please try again in 5 minutes.',
});

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'crop-image-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  // Accept only image files
  if (file.mimetype.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error('Only image files are allowed!'), false);
  }
};

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: fileFilter
});

// Model classes data (from training)
const CROP_CLASSES = [
  'Apple', 'Bell Pepper', 'Cherry', 'Citrus', 'Grape',
  'Maize', 'Cassava', 'Peach', 'Potato', 'Rice',
  'Soybean', 'Strawberry', 'Tomato'
];

const DISEASE_CLASSES = [
  'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
  'Bell_pepper___Bacterial_spot', 'Bell_pepper___healthy',
  'Cherry___Powdery_mildew', 'Cherry___healthy',
  'Citrus___Citrus_canker', 'Citrus___healthy',
  'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
  'Maize___Common_rust', 'Maize___Northern_Leaf_Blight', 'Maize___healthy',
  'Cassava___Bacterial_blight', 'Cassava___Brown_streak_disease', 'Cassava___Green_mottle', 'Cassava___Mosaic_disease', 'Cassava___healthy',
  'Peach___Bacterial_spot', 'Peach___healthy',
  'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
  'Rice___Brown_spot', 'Rice___Hispa', 'Rice___Leaf_blast', 'Rice___healthy',
  'Soybean___healthy',
  'Strawberry___Leaf_scorch', 'Strawberry___healthy',
  'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
  'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
];

// Treatment recommendations database
const TREATMENT_DATABASE = {
  'Apple___Apple_scab': {
    severity: 'Moderate',
    recommendations: [
      'Apply copper-based fungicide during early season',
      'Rake and destroy fallen leaves in autumn',
      'Prune for better air circulation',
      'Choose resistant varieties for new plantings'
    ],
    treatment: {
      chemical: 'Copper hydroxide or Myclobutanil',
      organic: 'Baking soda spray or sulfur-based fungicides',
      cultural: 'Sanitation, pruning, resistant varieties'
    }
  },
  'Tomato___Early_blight': {
    severity: 'Moderate',
    recommendations: [
      'Remove affected leaves immediately to prevent spread',
      'Apply copper-based fungicide every 7-14 days',
      'Improve air circulation around plants',
      'Avoid overhead watering to reduce leaf moisture',
      'Consider resistant varieties for future planting'
    ],
    treatment: {
      chemical: 'Copper hydroxide or Mancozeb',
      organic: 'Baking soda spray (1 tsp per quart water)',
      cultural: 'Crop rotation, proper spacing, mulching'
    }
  },
  'Maize___Common_rust': {
    severity: 'Low to Moderate',
    recommendations: [
      'Monitor crop regularly during humid conditions',
      'Apply fungicide if infection is severe',
      'Plant resistant hybrid varieties',
      'Ensure proper plant spacing for air circulation'
    ],
    treatment: {
      chemical: 'Propiconazole or Tebuconazole',
      organic: 'Neem oil or copper soap',
      cultural: 'Resistant varieties, crop rotation'
    }
  },
  // Add more treatments as needed...
};

// Real AI prediction function using Python inference script
const predictCropDisease = async (imagePath) => {
  return new Promise((resolve, reject) => {
    // Use virtual environment Python if available
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const venvPython = path.join(__dirname, 'venv', 'Scripts', 'python.exe');

    // Check if virtual environment exists, otherwise use system Python
    const pythonExecutable = fs.existsSync(venvPython) ? venvPython : pythonCmd;

    const pythonProcess = spawn(pythonExecutable, [
      path.join(__dirname, 'inference.py'),
      imagePath,
      path.join(__dirname, 'models')
    ], {
      cwd: __dirname,  // Run from backend directory
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process stderr:', stderr);
        reject(new Error(`Python inference failed: ${stderr || 'Unknown error'}`));
        return;
      }

      try {
        const result = JSON.parse(stdout.trim());
        if (result.success) {
          resolve(result.prediction);
        } else {
          reject(new Error(result.error || 'Prediction failed'));
        }
      } catch (parseError) {
        console.error('Failed to parse Python output:', stdout);
        reject(new Error('Invalid response from inference script'));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      reject(new Error('Failed to start inference process'));
    });
  });
};

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'Africa Crop Doctor AI Backend API',
    version: '1.0.0',
    status: 'active',
    endpoints: {
      health: '/api/health',
      predict: '/api/predict (POST)',
      model_info: '/api/model/info'
    }
  });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Model information endpoint
app.get('/api/model/info', (req, res) => {
  res.json({
    model: {
      name: 'Africa Crop Doctor AI',
      version: '1.0.0',
      architecture: 'EfficientNet-B0 with Attention',
      parameters: '5.3M',
      accuracy: {
        disease: 85.6,
        crop: 92.3
      },
      classes: {
        crops: CROP_CLASSES.length,
        diseases: DISEASE_CLASSES.length,
        total: DISEASE_CLASSES.length
      },
      supported_formats: ['jpg', 'jpeg', 'png', 'webp'],
      max_file_size: '10MB',
      input_size: '224x224'
    },
    training: {
      epochs: 50,
      dataset_size: 57050,
      datasets_used: 3,
      training_time: '4h 23m'
    }
  });
});

// Main prediction endpoint
app.post('/api/predict', predictionLimiter, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No image file provided',
        message: 'Please upload an image file'
      });
    }

    const imagePath = req.file.path;

    // Preprocess image
    const processedImagePath = path.join(uploadsDir, 'processed-' + req.file.filename);
    
    await sharp(imagePath)
      .resize(224, 224)
      .jpeg({ quality: 90 })
      .toFile(processedImagePath);

    // Get prediction from model
    const prediction = await predictCropDisease(processedImagePath);
    
    // Get treatment recommendations
    const treatmentInfo = TREATMENT_DATABASE[prediction.fullClass] || {
      severity: 'Unknown',
      recommendations: ['Consult with a local agricultural extension officer'],
      treatment: {
        chemical: 'Consult agricultural expert',
        organic: 'Maintain good crop hygiene',
        cultural: 'Follow best farming practices'
      }
    };

    // Format response
    const response = {
      prediction: {
        crop: prediction.crop,
        disease: prediction.disease.replace(/_/g, ' '),
        confidence: Math.round(prediction.confidence * 100) / 100,
        status: prediction.disease.toLowerCase().includes('healthy') ? 'healthy' : 'diseased',
        severity: treatmentInfo.severity
      },
      recommendations: treatmentInfo.recommendations,
      treatment: treatmentInfo.treatment,
      metadata: {
        model_version: '1.0.0',
        processing_time: Date.now() - req.file.uploadTime || 0,
        image_size: req.file.size,
        timestamp: new Date().toISOString()
      }
    };

    // Clean up uploaded files
    setTimeout(() => {
      [imagePath, processedImagePath].forEach(file => {
        fs.unlink(file, (err) => {
          if (err && err.code !== 'ENOENT') {
            console.warn(`Failed to delete file ${file}:`, err.message);
          }
        });
      });
    }, 5000); // Delete after 5 seconds

    res.json(response);

  } catch (error) {
    console.error('Prediction error:', error);
    
    // Clean up files on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      error: 'Prediction failed',
      message: error.message || 'Internal server error during prediction'
    });
  }
});

// Get supported classes
app.get('/api/classes', (req, res) => {
  res.json({
    crops: CROP_CLASSES,
    diseases: DISEASE_CLASSES.map(cls => ({
      full_name: cls,
      crop: cls.split('___')[0],
      disease: cls.split('___')[1]
    }))
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        error: 'File too large',
        message: 'Image file must be smaller than 10MB'
      });
    }
  }
  
  if (error.message === 'Only image files are allowed!') {
    return res.status(400).json({
      error: 'Invalid file type',
      message: 'Please upload a valid image file (JPG, PNG, WebP)'
    });
  }

  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'production' ? 'Something went wrong' : error.message
  });
});

// Handle 404
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    message: `The endpoint ${req.originalUrl} does not exist`
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Africa Crop Doctor AI Backend running on port ${PORT}`);
  console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🤖 Model: Real PyTorch inference enabled`);
  console.log(`🔗 API Documentation: http://localhost:${PORT}`);
  console.log(`💚 Health Check: http://localhost:${PORT}/api/health`);
});

module.exports = app;