import React from "react";

const Header = () => {
  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-indigo-600 rounded-lg flex items-center justify-center">
              <span className="text-2xl">👋</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                Sign Language Recognition
              </h1>
              <p className="text-gray-600 text-sm">
                Real-time gesture translation powered by AI
              </p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
