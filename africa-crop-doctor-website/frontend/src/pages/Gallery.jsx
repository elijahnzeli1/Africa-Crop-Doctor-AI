import React, { useState } from 'react';
import { 
  Image, 
  Search, 
  Filter, 
  Grid,
  List,
  Eye,
  Tag,
  Leaf
} from 'lucide-react';

const Gallery = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState('grid');

  // Sample images data - In a real app, this would come from your dataset
  const sampleImages = [
    // Healthy samples
    { id: 1, crop: 'Tomato', disease: 'Healthy', category: 'healthy', image: 'https://picsum.photos/300/300?random=1' },
    { id: 2, crop: 'Maize', disease: 'Healthy', category: 'healthy', image: 'https://picsum.photos/300/300?random=2' },
    { id: 3, crop: 'Cassava', disease: 'Healthy', category: 'healthy', image: 'https://picsum.photos/300/300?random=3' },
    
    // Disease samples
    { id: 4, crop: 'Tomato', disease: 'Early Blight', category: 'diseased', image: 'https://picsum.photos/300/300?random=4' },
    { id: 5, crop: 'Tomato', disease: 'Late Blight', category: 'diseased', image: 'https://picsum.photos/300/300?random=5' },
    { id: 6, crop: 'Maize', disease: 'Common Rust', category: 'diseased', image: 'https://picsum.photos/300/300?random=6' },
    { id: 7, crop: 'Maize', disease: 'Northern Leaf Blight', category: 'diseased', image: 'https://picsum.photos/300/300?random=7' },
    { id: 8, crop: 'Cassava', disease: 'Mosaic Disease', category: 'diseased', image: 'https://picsum.photos/300/300?random=8' },
    { id: 9, crop: 'Rice', disease: 'Brown Spot', category: 'diseased', image: 'https://picsum.photos/300/300?random=9' },
    { id: 10, crop: 'Potato', disease: 'Early Blight', category: 'diseased', image: 'https://picsum.photos/300/300?random=10' },
    { id: 11, crop: 'Bell Pepper', disease: 'Bacterial Spot', category: 'diseased', image: 'https://picsum.photos/300/300?random=11' },
    { id: 12, crop: 'Grape', disease: 'Black Rot', category: 'diseased', image: 'https://picsum.photos/300/300?random=12' },
    { id: 13, crop: 'Apple', disease: 'Apple Scab', category: 'diseased', image: 'https://picsum.photos/300/300?random=13' },
    { id: 14, crop: 'Cherry', disease: 'Powdery Mildew', category: 'diseased', image: 'https://picsum.photos/300/300?random=14' },
    { id: 15, crop: 'Strawberry', disease: 'Leaf Scorch', category: 'diseased', image: 'https://picsum.photos/300/300?random=15' },
    { id: 16, crop: 'Citrus', disease: 'Citrus Canker', category: 'diseased', image: 'https://picsum.photos/300/300?random=16' },
  ];

  const categories = [
    { value: 'all', label: 'All Images', count: sampleImages.length },
    { value: 'healthy', label: 'Healthy', count: sampleImages.filter(img => img.category === 'healthy').length },
    { value: 'diseased', label: 'Diseased', count: sampleImages.filter(img => img.category === 'diseased').length },
  ];

  const crops = [...new Set(sampleImages.map(img => img.crop))];

  const filteredImages = sampleImages.filter(image => {
    const matchesCategory = selectedCategory === 'all' || image.category === selectedCategory;
    const matchesSearch = searchTerm === '' || 
      image.crop.toLowerCase().includes(searchTerm.toLowerCase()) ||
      image.disease.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const ImageCard = ({ image }) => (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow group">
      <div className="relative aspect-square">
        <img 
          src={image.image} 
          alt={`${image.crop} - ${image.disease}`}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
        />
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center">
          <div className="opacity-0 group-hover:opacity-100 transition-opacity">
            <button className="bg-white rounded-full p-2 shadow-lg">
              <Eye className="w-5 h-5 text-gray-700" />
            </button>
          </div>
        </div>
        <div className="absolute top-3 right-3">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            image.category === 'healthy' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {image.category === 'healthy' ? 'Healthy' : 'Diseased'}
          </span>
        </div>
      </div>
      <div className="p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Leaf className="w-4 h-4 text-green-600" />
          <span className="font-semibold text-gray-900">{image.crop}</span>
        </div>
        <div className="flex items-center space-x-2">
          <Tag className="w-4 h-4 text-gray-500" />
          <span className="text-gray-600">{image.disease}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center">
              <Image className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Training Dataset Gallery</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Explore sample images from our comprehensive training dataset featuring 35 crop diseases 
            across 13 different crop types used to train the Africa Crop Doctor AI model.
          </p>
        </div>

        {/* Stats Bar */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-green-600">57,050+</div>
              <div className="text-gray-600">Total Images</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600">13</div>
              <div className="text-gray-600">Crop Types</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-600">35</div>
              <div className="text-gray-600">Disease Classes</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-red-600">3</div>
              <div className="text-gray-600">Datasets Used</div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search crops or diseases..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
            </div>

            {/* Category Filter */}
            <div className="flex items-center space-x-4">
              <Filter className="text-gray-500 w-5 h-5" />
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                {categories.map(category => (
                  <option key={category.value} value={category.value}>
                    {category.label} ({category.count})
                  </option>
                ))}
              </select>
            </div>

            {/* View Mode */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg ${viewMode === 'grid' ? 'bg-green-100 text-green-600' : 'text-gray-500 hover:bg-gray-100'}`}
              >
                <Grid className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-lg ${viewMode === 'list' ? 'bg-green-100 text-green-600' : 'text-gray-500 hover:bg-gray-100'}`}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Crop Filter Pills */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2">
            <button className="px-4 py-2 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              All Crops
            </button>
            {crops.slice(0, 6).map(crop => (
              <button key={crop} className="px-4 py-2 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 transition-colors">
                {crop}
              </button>
            ))}
            {crops.length > 6 && (
              <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 transition-colors">
                +{crops.length - 6} more
              </button>
            )}
          </div>
        </div>

        {/* Image Grid */}
        <div className={`grid gap-6 mb-8 ${
          viewMode === 'grid' 
            ? 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4' 
            : 'grid-cols-1'
        }`}>
          {filteredImages.map(image => (
            <ImageCard key={image.id} image={image} />
          ))}
        </div>

        {/* Load More Button */}
        <div className="text-center">
          <button className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold">
            Load More Images
          </button>
        </div>

        {/* Dataset Sources */}
        <div className="bg-white rounded-xl shadow-lg p-8 mt-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Dataset Sources</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">PlantDoc Dataset</h3>
              <p className="text-gray-600 mb-4">Comprehensive plant disease dataset with high-quality labeled images</p>
              <div className="text-sm text-gray-500">
                <div>• 2,569 images</div>
                <div>• 13 plant species</div>
                <div>• 17 disease classes</div>
              </div>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">PlantDisease Dataset</h3>
              <p className="text-gray-600 mb-4">Large-scale dataset focusing on leaf-based disease identification</p>
              <div className="text-sm text-gray-500">
                <div>• 54,305 images</div>
                <div>• 14 crop species</div>
                <div>• 26 disease classes</div>
              </div>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Custom Collection</h3>
              <p className="text-gray-600 mb-4">Curated images specific to African agricultural conditions</p>
              <div className="text-sm text-gray-500">
                <div>• 176 images</div>
                <div>• African crops focus</div>
                <div>• Field conditions</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Gallery;