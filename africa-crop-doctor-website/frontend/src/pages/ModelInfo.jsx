import React from 'react';
import { 
  Brain, 
  Layers, 
  Zap, 
  Target, 
  Database,
  Code,
  Download,
  FileText,
  Settings,
  CheckCircle
} from 'lucide-react';

const ModelInfo = () => {
  const architectureDetails = [
    {
      component: "Backbone",
      description: "EfficientNet-B0",
      details: "Pre-trained CNN optimized for efficiency and accuracy"
    },
    {
      component: "Attention Mechanism",
      description: "Channel & Spatial Attention",
      details: "Enhanced feature extraction with attention weights"
    },
    {
      component: "Classifier Head",
      description: "Multi-layer Perceptron",
      details: "Dropout regularization and batch normalization"
    },
    {
      component: "Output Layer",
      description: "35 Disease Classes",
      details: "Softmax activation for probability distribution"
    }
  ];

  const technicalSpecs = [
    { label: "Total Parameters", value: "5.3M", icon: <Settings className="w-5 h-5" /> },
    { label: "Model Size", value: "20.2 MB", icon: <Database className="w-5 h-5" /> },
    { label: "Input Resolution", value: "224×224", icon: <Target className="w-5 h-5" /> },
    { label: "Inference Speed", value: "~50ms", icon: <Zap className="w-5 h-5" /> },
    { label: "Framework", value: "PyTorch", icon: <Code className="w-5 h-5" /> },
    { label: "Precision", value: "FP32", icon: <FileText className="w-5 h-5" /> }
  ];

  const modelFormats = [
    {
      format: "PyTorch (.pth)",
      size: "20.2 MB",
      description: "Native PyTorch format for training and fine-tuning",
      available: true
    },
    {
      format: "TorchScript (.pt)",
      size: "20.2 MB", 
      description: "Optimized for production deployment",
      available: true
    },
    {
      format: "ONNX (.onnx)",
      size: "20.2 MB",
      description: "Cross-platform inference format",
      available: true
    },
    {
      format: "SafeTensors (.safetensors)",
      size: "20.2 MB",
      description: "Secure tensor serialization format",
      available: true
    }
  ];

  const performanceMetrics = [
    { metric: "Disease Accuracy", value: "85.6%", description: "Overall disease classification accuracy" },
    { metric: "Crop Accuracy", value: "92.3%", description: "Crop type identification accuracy" },
    { metric: "Training Loss", value: "0.3847", description: "Final training loss after 50 epochs" },
    { metric: "Validation Loss", value: "0.4234", description: "Best validation loss achieved" }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center">
              <Brain className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Model Architecture</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Detailed technical information about the Africa Crop Doctor AI model, 
            its architecture, performance metrics, and available formats.
          </p>
        </div>

        {/* Performance Overview */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Target className="w-6 h-6 mr-3 text-green-600" />
            Performance Metrics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {performanceMetrics.map((metric, index) => (
              <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-3xl font-bold text-green-600 mb-2">{metric.value}</div>
                <div className="text-lg font-semibold text-gray-900 mb-1">{metric.metric}</div>
                <div className="text-sm text-gray-600">{metric.description}</div>
              </div>
            ))}
          </div>
          
          {/* Model Evaluation Results Chart */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">Detailed Evaluation Results</h3>
            <img 
              src="/images/model_evaluation_results.png" 
              alt="Model Evaluation Results - Confusion Matrix and Performance Metrics"
              className="w-full h-auto rounded-lg shadow-sm"
            />
            <p className="text-sm text-gray-600 mt-4 text-center">
              Comprehensive evaluation showing confusion matrix, precision, recall, and F1-scores across all 35 disease classes
            </p>
          </div>
        </div>

        {/* Architecture Details */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Layers className="w-6 h-6 mr-3 text-green-600" />
            Architecture Components
          </h2>
          <div className="space-y-6">
            {architectureDetails.map((component, index) => (
              <div key={index} className="border-l-4 border-green-500 pl-6 py-2">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">{component.component}</h3>
                  <span className="text-green-600 font-medium">{component.description}</span>
                </div>
                <p className="text-gray-600">{component.details}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Settings className="w-6 h-6 mr-3 text-green-600" />
            Technical Specifications
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {technicalSpecs.map((spec, index) => (
              <div key={index} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center text-green-600">
                  {spec.icon}
                </div>
                <div>
                  <div className="font-semibold text-gray-900">{spec.label}</div>
                  <div className="text-green-600 font-medium">{spec.value}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Formats */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Download className="w-6 h-6 mr-3 text-green-600" />
            Available Model Formats
          </h2>
          <div className="space-y-4">
            {modelFormats.map((format, index) => (
              <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-green-300 transition-colors">
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                    {format.available ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : (
                      <FileText className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">{format.format}</div>
                    <div className="text-sm text-gray-600">{format.description}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-gray-900">{format.size}</div>
                  <div className={`text-sm ${format.available ? 'text-green-600' : 'text-gray-400'}`}>
                    {format.available ? 'Available' : 'Coming Soon'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Training Details */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Brain className="w-6 h-6 mr-3 text-green-600" />
            Training Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization</h3>
              <ul className="space-y-2 text-gray-600">
                <li><strong>Optimizer:</strong> AdamW with weight decay</li>
                <li><strong>Learning Rate:</strong> OneCycleLR scheduler</li>
                <li><strong>Max LR:</strong> 3e-4</li>
                <li><strong>Batch Size:</strong> 32</li>
                <li><strong>Epochs:</strong> 50</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Regularization</h3>
              <ul className="space-y-2 text-gray-600">
                <li><strong>Label Smoothing:</strong> 0.1</li>
                <li><strong>Weight Decay:</strong> 1e-4</li>
                <li><strong>Dropout:</strong> 0.5</li>
                <li><strong>Data Augmentation:</strong> Advanced transforms</li>
                <li><strong>Mixed Precision:</strong> Enabled</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;