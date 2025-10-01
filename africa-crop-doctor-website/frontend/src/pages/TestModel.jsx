import React, { useState } from 'react';
import { 
  Upload, 
  Camera, 
  Loader, 
  CheckCircle, 
  AlertTriangle,
  Eye,
  Download,
  RefreshCw,
  Sparkles
} from 'lucide-react';

const TestModel = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);  
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleImageUpload = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setPrediction(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  };

  const runPrediction = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    
    try {
      // Create FormData for image upload
      const formData = new FormData();
      formData.append('image', selectedImage);

      // Make API call to backend
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Prediction failed');
      }

      const result = await response.json();
      
      // Format the response for the UI
      const formattedPrediction = {
        crop: result.prediction.crop,
        disease: result.prediction.disease,
        confidence: result.prediction.confidence,
        severity: result.prediction.severity,
        status: result.prediction.status,
        recommendations: result.recommendations,
        treatment: result.treatment,
        metadata: result.metadata
      };

      setPrediction(formattedPrediction);
    } catch (error) {
      console.error('Prediction failed:', error);
      // Show error to user
      setPrediction({
        error: true,
        message: error.message || 'Failed to analyze image. Please try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPrediction(null);
  };

  const sampleImages = [
    { src: 'https://picsum.photos/150/150?random=101', name: 'Tomato Early Blight' },
    { src: 'https://picsum.photos/150/150?random=102', name: 'Maize Common Rust' },
    { src: 'https://picsum.photos/150/150?random=103', name: 'Cassava Mosaic' },
    { src: 'https://picsum.photos/150/150?random=104', name: 'Healthy Tomato' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Test the AI Model</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload an image of your crop to get instant disease detection results with 85.6% accuracy. 
            Our AI will identify the crop type, detect any diseases, and provide treatment recommendations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            {/* Image Upload Area */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Crop Image</h2>
              
              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                  dragActive 
                    ? 'border-green-500 bg-green-50' 
                    : 'border-gray-300 hover:border-green-400'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Preview" 
                      className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
                    />
                    <div className="flex justify-center space-x-4">
                      <button
                        onClick={runPrediction}
                        disabled={isLoading}
                        className="flex items-center px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {isLoading ? (
                          <>
                            <Loader className="w-4 h-4 mr-2 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Eye className="w-4 h-4 mr-2" />
                            Analyze Image
                          </>
                        )}
                      </button>
                      <button
                        onClick={reset}
                        className="flex items-center px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Reset
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                      <Upload className="w-8 h-8 text-green-600" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-gray-900 mb-2">
                        Drag and drop your image here
                      </p>
                      <p className="text-gray-600 mb-4">or click to browse</p>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileInputChange}
                        className="hidden"
                        id="image-upload"
                      />
                      <label
                        htmlFor="image-upload"
                        className="inline-flex items-center px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 cursor-pointer transition-colors"
                      >
                        <Camera className="w-4 h-4 mr-2" />
                        Choose Image
                      </label>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 text-sm text-gray-500">
                <p>Supported formats: JPG, PNG, WebP</p>
                <p>Maximum file size: 10MB</p>
              </div>
            </div>

            {/* Sample Images */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Try Sample Images</h3>
              <div className="grid grid-cols-2 gap-4">
                {sampleImages.map((sample, index) => (
                  <button
                    key={index}
                    className="group relative overflow-hidden rounded-lg border-2 border-gray-200 hover:border-green-400 transition-colors"
                    onClick={() => {
                      // In a real app, you'd load the actual sample image
                      setImagePreview(sample.src);
                      // Create a mock file object
                      setSelectedImage(new File([], sample.name));
                    }}
                  >
                    <img 
                      src={sample.src} 
                      alt={sample.name}
                      className="w-full aspect-square object-cover group-hover:scale-105 transition-transform"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                      <span className="text-white font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                        Use Sample
                      </span>
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-2">
                      <p className="text-white text-xs font-medium">{sample.name}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {prediction ? (
              prediction.error ? (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <AlertTriangle className="w-6 h-6 text-red-600" />
                    <h2 className="text-xl font-semibold text-gray-900">Prediction Error</h2>
                  </div>
                  <p className="text-red-600 mb-4">{prediction.message}</p>
                  <button
                    onClick={reset}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              ) : (
                <>
                  {/* Main Result */}
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xl font-semibold text-gray-900">Detection Results</h2>
                      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
                        prediction.status === 'healthy' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {prediction.status === 'healthy' ? (
                          <CheckCircle className="w-4 h-4" />
                        ) : (
                          <AlertTriangle className="w-4 h-4" />
                        )}
                        <span className="font-medium capitalize">{prediction.status}</span>
                      </div>
                    </div>

                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-gray-700">Crop Type</label>
                      <p className="text-lg text-gray-900">{prediction.crop}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-gray-700">Disease Detected</label>
                      <p className="text-lg text-gray-900">{prediction.disease}</p>
                    </div>

                    <div>
                      <label className="text-sm font-medium text-gray-700">Confidence Level</label>
                      <div className="mt-1">
                        <div className="flex items-center space-x-3">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${prediction.confidence}%` }}
                            ></div>
                          </div>
                          <span className="text-lg font-semibold text-green-600">
                            {prediction.confidence}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {prediction.severity && (
                      <div>
                        <label className="text-sm font-medium text-gray-700">Severity</label>
                        <p className="text-lg text-gray-900">{prediction.severity}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Treatment Recommendations */}
                {prediction.recommendations && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Treatment Recommendations</h3>
                    <ul className="space-y-2">
                      {prediction.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start space-x-3">
                          <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-700">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Treatment Options */}
                {prediction.treatment && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Treatment Options</h3>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium text-gray-900 mb-1">Chemical Treatment</h4>
                        <p className="text-gray-700 text-sm">{prediction.treatment.chemical}</p>
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900 mb-1">Organic Treatment</h4>
                        <p className="text-gray-700 text-sm">{prediction.treatment.organic}</p>
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900 mb-1">Cultural Practices</h4>
                        <p className="text-gray-700 text-sm">{prediction.treatment.cultural}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Download Results */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Export Results</h3>
                  <div className="flex space-x-4">
                    <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                      <Download className="w-4 h-4 mr-2" />
                      Download PDF Report
                    </button>
                    <button className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
                      <Download className="w-4 h-4 mr-2" />
                      Export JSON
                    </button>
                  </div>
                </div>
                </>
              )
            ) : (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready to Analyze</h3>
                <p className="text-gray-600">
                  Upload an image to get instant crop disease detection results with treatment recommendations.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Model Info */}
        <div className="mt-12 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Upload className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">1. Upload Image</h3>
              <p className="text-gray-600">Upload a clear photo of your crop leaves showing any symptoms</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Sparkles className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">2. AI Analysis</h3>
              <p className="text-gray-600">Our advanced AI model analyzes the image using deep learning</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">3. Get Results</h3>
              <p className="text-gray-600">Receive instant diagnosis with treatment recommendations</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestModel;