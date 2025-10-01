import React from 'react';
import { 
  Leaf, 
  Brain, 
  Eye, 
  Shield, 
  TrendingUp, 
  Users, 
  Globe,
  CheckCircle,
  ArrowRight
} from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "Advanced AI Technology",
      description: "Powered by EfficientNet-B0 with attention mechanisms for superior disease detection"
    },
    {
      icon: <Eye className="w-8 h-8" />,
      title: "High Accuracy",
      description: "85.6% disease classification accuracy across 35 different crop diseases"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Reliable Detection",
      description: "Trained on 57,050+ images from multiple verified agricultural datasets"
    },
    {
      icon: <Globe className="w-8 h-8" />,
      title: "Africa-Focused",
      description: "Specifically designed for African crops and agricultural conditions"
    }
  ];

  const stats = [
    { number: "35", label: "Disease Classes", icon: <TrendingUp className="w-6 h-6" /> },
    { number: "13", label: "Crop Types", icon: <Leaf className="w-6 h-6" /> },
    { number: "57K+", label: "Training Images", icon: <Eye className="w-6 h-6" /> },
    { number: "85.6%", label: "Accuracy", icon: <CheckCircle className="w-6 h-6" /> }
  ];

  const supportedCrops = [
    "Maize", "Cassava", "Rice", "Tomato", "Potato", "Bell Pepper",
    "Grape", "Apple", "Cherry", "Peach", "Strawberry", "Citrus", "Soybean"
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-green-50 to-blue-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center">
                <Leaf className="w-12 h-12 text-white" />
              </div>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              Africa Crop Doctor <span className="text-green-600">AI</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Revolutionary AI-powered crop disease detection system designed specifically for African agriculture. 
              Identify diseases instantly with 85.6% accuracy and help farmers protect their harvests.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="/test" 
                className="inline-flex items-center px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold"
              >
                Test the Model
                <ArrowRight className="ml-2 w-5 h-5" />
              </a>
              <a 
                href="/model" 
                className="inline-flex items-center px-8 py-3 border-2 border-green-600 text-green-600 rounded-lg hover:bg-green-50 transition-colors font-semibold"
              >
                Learn More
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="flex justify-center mb-4">
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-green-600">
                    {stat.icon}
                  </div>
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.number}</div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Advanced AI Technology for Agriculture
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our state-of-the-art model combines cutting-edge deep learning with agricultural expertise
              to deliver accurate, reliable crop disease detection.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-green-600 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>

          {/* African Focus Section */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center mb-8">
              <h3 className="text-3xl font-bold text-gray-900 mb-4">Designed for African Agriculture</h3>
              <p className="text-lg text-gray-600">
                Our AI model is specifically trained and optimized for African crop varieties and agricultural conditions
              </p>
            </div>
            <div className="flex justify-center">
              <img 
                src="/images/map.jpeg" 
                alt="Africa Map - Geographic Coverage of Africa Crop Doctor AI"
                className="max-w-full h-auto rounded-lg shadow-lg"
              />
            </div>
            <div className="mt-6 text-center">
              <p className="text-gray-600">
                Supporting farmers across Africa with localized crop disease detection and treatment recommendations
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Crops Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Supported Crops
            </h2>
            <p className="text-xl text-gray-600">
              Our AI model is trained to detect diseases across 13 major crop types
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {supportedCrops.map((crop, index) => (
              <div key={index} className="bg-gray-50 rounded-lg p-4 text-center hover:bg-green-50 transition-colors">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Leaf className="w-4 h-4 text-green-600" />
                </div>
                <span className="text-gray-700 font-medium">{crop}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 gradient-bg">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            Ready to Test Our AI Model?
          </h2>
          <p className="text-xl text-green-100 mb-8">
            Upload an image of your crop and get instant disease detection results with treatment recommendations.
          </p>
          <a 
            href="/test" 
            className="inline-flex items-center px-8 py-3 bg-white text-green-600 rounded-lg hover:bg-gray-100 transition-colors font-semibold text-lg"
          >
            Start Testing Now
            <ArrowRight className="ml-2 w-5 h-5" />
          </a>
        </div>
      </section>
    </div>
  );
};

export default Home;