"use client";

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sparkles, Download, Image, Scissors, Wand2 } from "lucide-react";

export default function HomePage() {
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState("");

  const API_BASE = "http://localhost:8000";

  const handleProcess = async () => {
    if (!youtubeUrl) return setError("Please enter a YouTube URL");

    setError("");
    setLoading(true);
    setResults(null);

    try {
      const res = await fetch(`${API_BASE}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: youtubeUrl }),
      });

      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError("Failed to process video");
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = (url: string, filename: string) => {
    fetch(url)
      .then((res) => res.blob())
      .then((blob) => {
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
      });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-6 md:p-10">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl mb-4 shadow-lg">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3">
            AI Product Imagery
          </h1>
          <p className="text-gray-600 text-lg">
            Extract and enhance product images from YouTube videos
          </p>
        </div>

        {/* Input Section */}
        <Card className="p-6 md:p-8 shadow-xl border-0 bg-white/80 backdrop-blur-sm mb-8">
          <div className="flex flex-col md:flex-row gap-3">
            <Input
              placeholder="Paste YouTube URL here..."
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              className="text-lg h-12 border-2 border-gray-200 focus:border-indigo-500 transition-colors"
              onKeyPress={(e) => e.key === "Enter" && handleProcess()}
            />
            <Button 
              onClick={handleProcess} 
              className="h-12 px-8 text-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all"
              disabled={loading}
            >
              {loading ? "Processing..." : "Process"}
            </Button>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg">
              <p className="text-red-700 font-medium">{error}</p>
            </div>
          )}
        </Card>

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="relative">
              <div className="w-20 h-20 border-4 border-indigo-200 rounded-full"></div>
              <div className="w-20 h-20 border-4 border-indigo-600 rounded-full animate-spin border-t-transparent absolute top-0 left-0"></div>
            </div>
            <p className="mt-6 text-gray-600 font-medium">Processing your video...</p>
          </div>
        )}

        {/* Results */}
        {!loading && results?.products && (
          <div className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-8 w-1 bg-gradient-to-b from-indigo-600 to-purple-600 rounded-full"></div>
              <h2 className="text-3xl font-bold text-gray-800">
                Products Identified
                <span className="ml-3 text-lg font-normal text-gray-500">
                  ({results.products.length})
                </span>
              </h2>
            </div>

            {results.products.map((p: any, index: number) => {
              const label = p.label.charAt(0).toUpperCase() + p.label.slice(1);
              const bestFrameURL = `${API_BASE}/files?path=${encodeURIComponent(p.best_frame)}`;
              const croppedURL = `${API_BASE}/files?path=${encodeURIComponent(p.cropped)}`;

              return (
                <Card
                  key={index}
                  className="p-6 md:p-8 shadow-xl border-0 bg-white/90 backdrop-blur-sm hover:shadow-2xl transition-all duration-300"
                >
                  <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-100">
                    <div className="w-10 h-10 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-lg flex items-center justify-center">
                      <span className="text-xl font-bold text-indigo-600">
                        {index + 1}
                      </span>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-800">{label}</h3>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Best Frame */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 mb-3">
                        <Image className="w-5 h-5 text-indigo-600" />
                        <p className="font-semibold text-gray-700">Best Frame</p>
                      </div>
                      <div className="relative group overflow-hidden rounded-xl shadow-lg">
                        <img
                          src={bestFrameURL}
                          className="w-full h-48 object-cover group-hover:scale-110 transition-transform duration-300"
                          alt={`${label} best frame`}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full border-2 hover:bg-indigo-50 hover:border-indigo-300 transition-colors"
                        onClick={() => downloadImage(bestFrameURL, `${label}-best-frame.jpg`)}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    </div>

                    {/* Segmented */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 mb-3">
                        <Scissors className="w-5 h-5 text-purple-600" />
                        <p className="font-semibold text-gray-700">Segmented</p>
                      </div>
                      <div className="relative group overflow-hidden rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-gray-100">
                        <img
                          src={croppedURL}
                          className="w-full h-48 object-contain p-4 group-hover:scale-110 transition-transform duration-300"
                          alt={`${label} segmented`}
                        />
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full border-2 hover:bg-purple-50 hover:border-purple-300 transition-colors"
                        onClick={() => downloadImage(croppedURL, `${label}-segmented.png`)}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    </div>

                    {/* Enhanced Images */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 mb-3">
                        <Wand2 className="w-5 h-5 text-pink-600" />
                        <p className="font-semibold text-gray-700">Enhanced Images</p>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        {p.enhanced.map((img: string, i: number) => {
                          const enhURL = `${API_BASE}/files?path=${encodeURIComponent(img)}`;
                          return (
                            <div key={i} className="space-y-2">
                              <div className="relative group overflow-hidden rounded-lg shadow-md">
                                <img
                                  src={enhURL}
                                  className="w-full h-20 object-cover group-hover:scale-110 transition-transform duration-300"
                                  alt={`${label} enhanced ${i + 1}`}
                                />
                              </div>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="w-full h-8 text-xs hover:bg-pink-50 transition-colors"
                                onClick={() => downloadImage(enhURL, `${label}-enhanced-${i + 1}.jpg`)}
                              >
                                <Download className="w-3 h-3 mr-1" />
                                DL
                              </Button>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}