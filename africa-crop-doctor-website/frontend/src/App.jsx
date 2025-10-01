import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import ModelInfo from './pages/ModelInfo';
import Gallery from './pages/Gallery';
import TestModel from './pages/TestModel';
import Training from './pages/Training';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/model" element={<ModelInfo />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/test" element={<TestModel />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;