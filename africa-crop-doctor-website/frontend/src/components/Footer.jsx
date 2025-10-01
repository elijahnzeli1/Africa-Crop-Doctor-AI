import React from 'react';
import { Leaf, Github, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Logo and Description */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center">
                <Leaf className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold">Africa Crop Doctor AI</h2>
                <p className="text-gray-400 text-sm">v1.0</p>
              </div>
            </div>
            <p className="text-gray-300 mb-4 max-w-md">
              Advanced AI-powered crop disease detection system designed specifically for African agriculture. 
              Helping farmers identify and treat crop diseases with 85.6% accuracy.
            </p>
            <div className="flex space-x-4">
              <a 
                href="https://github.com" 
                className="text-gray-400 hover:text-white transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Github className="w-5 h-5" />
              </a>
              <a 
                href="#" 
                className="text-gray-400 hover:text-white transition-colors"
              >
                <ExternalLink className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><a href="/model" className="text-gray-300 hover:text-white transition-colors">Model Architecture</a></li>
              <li><a href="/training" className="text-gray-300 hover:text-white transition-colors">Training Details</a></li>
              <li><a href="/gallery" className="text-gray-300 hover:text-white transition-colors">Image Gallery</a></li>
              <li><a href="/test" className="text-gray-300 hover:text-white transition-colors">Test Model</a></li>
            </ul>
          </div>

          {/* Technical Info */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Technical</h3>
            <ul className="space-y-2 text-gray-300">
              <li>EfficientNet-B0 Backbone</li>
              <li>5.3M Parameters</li>
              <li>85.6% Disease Accuracy</li>
              <li>35 Disease Classes</li>
              <li>13 Crop Types</li>
              <li>PyTorch Framework</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400">
            © 2025 Africa Crop Doctor AI. Created for agricultural innovation in Africa.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;