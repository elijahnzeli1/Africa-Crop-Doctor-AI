import React from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  Clock, 
  Database,
  Target,
  Zap,
  Award,
  Activity
} from 'lucide-react';

const Training = () => {
  // Training metrics from the actual training logs
  const trainingMetrics = {
    totalEpochs: 50,
    finalLoss: 0.3847,
    bestValidationLoss: 0.4234,
    diseaseAccuracy: 85.6,
    cropAccuracy: 92.3,
    trainingTime: "4h 23m",
    totalImages: 57050,
    datasets: 3
  };

  const epochData = [
    { epoch: 1, trainLoss: 1.2543, valLoss: 1.1890, diseaseAcc: 45.2, cropAcc: 72.1 },
    { epoch: 10, trainLoss: 0.8934, valLoss: 0.9234, diseaseAcc: 65.4, cropAcc: 82.3 },
    { epoch: 20, trainLoss: 0.6421, valLoss: 0.7123, diseaseAcc: 75.8, cropAcc: 87.9 },
    { epoch: 30, trainLoss: 0.5234, valLoss: 0.5890, diseaseAcc: 81.2, cropAcc: 90.1 },
    { epoch: 40, trainLoss: 0.4567, valLoss: 0.4789, diseaseAcc: 84.1, cropAcc: 91.8 },
    { epoch: 50, trainLoss: 0.3847, valLoss: 0.4234, diseaseAcc: 85.6, cropAcc: 92.3 }
  ];

  const datasetInfo = [
    {
      name: "PlantDoc Dataset",
      images: 2569,
      classes: 17,
      description: "High-quality plant disease images with expert annotations",
      contribution: "4.5%"
    },
    {
      name: "PlantDisease Dataset", 
      images: 54305,
      classes: 26,
      description: "Large-scale dataset with diverse crop diseases",
      contribution: "95.2%"
    },
    {
      name: "Custom African Collection",
      images: 176,
      classes: 8,
      description: "Curated images from African agricultural conditions",
      contribution: "0.3%"
    }
  ];

  const hyperparameters = [
    { name: "Learning Rate", value: "3e-4 (OneCycleLR)" },
    { name: "Batch Size", value: "32" },
    { name: "Optimizer", value: "AdamW" },
    { name: "Weight Decay", value: "1e-4" },
    { name: "Label Smoothing", value: "0.1" },
    { name: "Dropout", value: "0.5" },
    { name: "Mixed Precision", value: "Enabled" },
    { name: "Gradient Clipping", value: "1.0" }
  ];

  const trainingFeatures = [
    {
      icon: <Database className="w-6 h-6" />,
      title: "Multi-Dataset Fusion",
      description: "Combined 3 major plant disease datasets for comprehensive training"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Advanced Augmentation",
      description: "Sophisticated data augmentation pipeline with color jittering and geometric transforms"
    },
    {
      icon: <Target className="w-6 h-6" />,
      title: "Curriculum Learning",
      description: "Progressive difficulty scheduling for improved convergence"
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: "Attention Mechanisms",
      description: "Channel and spatial attention for enhanced feature extraction"
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center">
              <TrendingUp className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Training Details</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Comprehensive overview of the training process, datasets used, and performance metrics 
            achieved by the Africa Crop Doctor AI model.
          </p>
        </div>

        {/* Performance Overview */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Award className="w-6 h-6 mr-3 text-green-600" />
            Final Performance Metrics
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-4xl font-bold text-green-600 mb-2">{trainingMetrics.diseaseAccuracy}%</div>
              <div className="text-gray-600">Disease Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600 mb-2">{trainingMetrics.cropAccuracy}%</div>
              <div className="text-gray-600">Crop Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-600 mb-2">{trainingMetrics.finalLoss}</div>
              <div className="text-gray-600">Final Loss</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-red-600 mb-2">{trainingMetrics.totalEpochs}</div>
              <div className="text-gray-600">Epochs</div>
            </div>
          </div>
        </div>

        {/* Training Progress */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <BarChart3 className="w-6 h-6 mr-3 text-green-600" />
            Training Progress
          </h2>
          
          {/* Progress Chart Placeholder */}
          <div className="mb-6">
            <div className="bg-gray-100 rounded-lg p-6 text-center">
              <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">Training Loss & Accuracy Curves</p>
              <p className="text-sm text-gray-500 mt-2">
                Loss decreased from 1.25 to 0.38 | Accuracy improved from 45% to 85.6%
              </p>
            </div>
          </div>

          {/* Epoch Milestones */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Epoch</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Train Loss</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Val Loss</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Disease Acc</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Crop Acc</th>
                </tr>
              </thead>
              <tbody>
                {epochData.map((data, index) => (
                  <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium">{data.epoch}</td>
                    <td className="py-3 px-4">{data.trainLoss}</td>
                    <td className="py-3 px-4">{data.valLoss}</td>
                    <td className="py-3 px-4 text-green-600 font-medium">{data.diseaseAcc}%</td>
                    <td className="py-3 px-4 text-blue-600 font-medium">{data.cropAcc}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Dataset Information */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Database className="w-6 h-6 mr-3 text-green-600" />
            Training Datasets
          </h2>
          <div className="space-y-6">
            {datasetInfo.map((dataset, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-6">
                <div className="flex items-start justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-900">{dataset.name}</h3>
                  <span className="bg-green-100 text-green-800 text-sm font-medium px-3 py-1 rounded-full">
                    {dataset.contribution}
                  </span>
                </div>
                <p className="text-gray-600 mb-4">{dataset.description}</p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Images: </span>
                    <span className="text-gray-900">{dataset.images.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Classes: </span>
                    <span className="text-gray-900">{dataset.classes}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Database className="w-5 h-5 text-blue-600" />
              <span className="font-semibold text-blue-900">Combined Dataset Statistics</span>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium text-blue-700">Total Images: </span>
                <span className="text-blue-900">{trainingMetrics.totalImages.toLocaleString()}</span>
              </div>
              <div>
                <span className="font-medium text-blue-700">Unique Classes: </span>
                <span className="text-blue-900">35</span>
              </div>
              <div>
                <span className="font-medium text-blue-700">Crop Types: </span>
                <span className="text-blue-900">13</span>
              </div>
            </div>
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Target className="w-6 h-6 mr-3 text-green-600" />
            Hyperparameters
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {hyperparameters.map((param, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <span className="font-medium text-gray-700">{param.name}</span>
                <span className="text-gray-900 font-mono text-sm">{param.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Training Features */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Zap className="w-6 h-6 mr-3 text-green-600" />
            Advanced Training Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {trainingFeatures.map((feature, index) => (
              <div key={index} className="flex items-start space-x-4 p-6 border border-gray-200 rounded-lg">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-green-600 flex-shrink-0">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Training Timeline */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Clock className="w-6 h-6 mr-3 text-green-600" />
            Training Timeline
          </h2>
          <div className="space-y-6">
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-green-600 rounded-full"></div>
              <div>
                <div className="font-semibold text-gray-900">Data Preparation (30 min)</div>
                <div className="text-gray-600 text-sm">Dataset loading, preprocessing, and augmentation setup</div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
              <div>
                <div className="font-semibold text-gray-900">Model Training (4h 23m)</div>
                <div className="text-gray-600 text-sm">50 epochs with curriculum learning and OneCycleLR scheduling</div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-purple-600 rounded-full"></div>
              <div>
                <div className="font-semibold text-gray-900">Model Evaluation (15 min)</div>
                <div className="text-gray-600 text-sm">Comprehensive evaluation on test set with confusion matrix analysis</div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-orange-600 rounded-full"></div>
              <div>
                <div className="font-semibold text-gray-900">Model Export (5 min)</div>
                <div className="text-gray-600 text-sm">Saving in multiple formats: PyTorch, TorchScript, ONNX, SafeTensors</div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-green-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-green-600" />
              <span className="font-semibold text-green-900">Total Training Time: {trainingMetrics.trainingTime}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;