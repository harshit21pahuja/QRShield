import React, { useState, useEffect, useRef } from 'react';
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  UploadCloud,
  Eye,
  Activity,
  Lock,
  CheckCircle2,
  XCircle,
  Copy,
  Check,
  Zap,
  RefreshCw,
  Layers,
  Image as ImageIcon
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState('url'); // 'url' | 'vision'
  const fileInputRef = useRef(null);

  // Health check on component mount
  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!isMounted) return;
        if (res.ok) {
          const data = await res.json();
          setHealthStatus(data);
        } else {
          setHealthStatus({ status: 'degraded' });
        }
      } catch {
        if (isMounted) {
          setHealthStatus({ status: 'offline' });
        }
      }
    };

    fetchHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleFileChange = (file) => {
    if (!file) return;

    // Client-side file size validation (10 MB limit)
    if (file.size > MAX_UPLOAD_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      setError(`File size (${sizeMb} MB) exceeds maximum upload limit of 10 MB. Please upload a smaller image.`);
      setSelectedFile(null);
      setPreviewUrl(null);
      setResult(null);
      return;
    }

    // Client-side format validation
    const nameLower = file.name.toLowerCase();
    const hasValidExt = ALLOWED_IMAGE_EXTENSIONS.some((ext) => nameLower.endsWith(ext));
    const hasValidType = file.type ? file.type.startsWith('image/') : false;

    if (!hasValidExt && !hasValidType) {
      setError(`Unsupported file type '${file.type || file.name}'. Please upload a PNG, JPG, or WEBP image.`);
      setSelectedFile(null);
      setPreviewUrl(null);
      setResult(null);
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const loadSample = async (samplePath, filename) => {
    try {
      const res = await fetch(samplePath);
      if (!res.ok) {
        throw new Error(`Sample file '${filename}' could not be fetched.`);
      }
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || 'image/png' });
      handleFileChange(file);
    } catch (err) {
      setError(`Failed to load sample: ${err.message}`);
    }
  };

  const handleScan = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000); // 25s timeout

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      let data;
      try {
        data = await response.json();
      } catch {
        data = { status: 'error', error: `Server returned HTTP ${response.status}` };
      }

      if (response.status === 413) {
        setError('File exceeds maximum upload size limit (10 MB). Please choose a smaller image.');
      } else if (!response.ok || data.status === 'error') {
        setError(data.error || 'Failed to analyze QR code.');
      } else {
        setResult(data);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Analysis timed out after 25 seconds. Please try again with a clearer QR image.');
      } else {
        setError(`Cannot reach QRShield backend at ${API_BASE}. Make sure the FastAPI service is running.`);
      }
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isMalicious = result?.prediction === 'MALICIOUS';
  const riskPercent = result ? Math.round(result.calibrated_probability * 100) : 0;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <header
        style={{
          borderBottom: '1px solid var(--border-color)',
          background: 'rgba(10, 13, 20, 0.85)',
          backdropFilter: 'blur(12px)',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          padding: '16px 24px',
        }}
      >
        <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(99, 102, 241, 0.2))',
                border: '1px solid rgba(6, 182, 212, 0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-cyan)',
              }}
            >
              <Shield size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="font-display" style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                  QR<span className="gradient-text-cyan">Shield</span>
                </span>
                <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>PROTOTYPE V1.0</span>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Explainable Multimodal Quishing Detection
              </p>
            </div>
          </div>

          {/* System Status Indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div
              id="health-badge"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 14px',
                borderRadius: '9999px',
                background: healthStatus?.status === 'healthy' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${healthStatus?.status === 'healthy' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                fontSize: '0.8rem',
              }}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: healthStatus?.status === 'healthy' ? '#10b981' : '#ef4444',
                }}
              />
              <span style={{ color: healthStatus?.status === 'healthy' ? '#34d399' : '#f87171' }}>
                {healthStatus?.status === 'healthy' ? 'Models Ready (Local)' : 'Backend Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Safety Notice Banner */}
      <div
        style={{
          background: 'rgba(6, 182, 212, 0.05)',
          borderBottom: '1px solid rgba(6, 182, 212, 0.15)',
          padding: '10px 24px',
          fontSize: '0.82rem',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
        }}
      >
        <Lock size={14} color="var(--accent-cyan)" />
        <span>
          <strong style={{ color: 'var(--accent-cyan)' }}>Zero-Contact Safety Guarantee:</strong> Scanned QR URLs are strictly analyzed as inert text. QRShield never visits, requests, redirects to, or executes destination links.
        </span>
      </div>

      {/* Main Container */}
      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px', flex: 1, width: '100%' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 420px) 1fr', gap: '32px' }}>
          
          {/* Left Column: Upload & Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h2 className="font-display" style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <UploadCloud size={20} color="var(--accent-cyan)" />
                Upload QR Code
              </h2>

              {/* Drag & Drop Area */}
              <div
                id="dropzone"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed var(--border-color)',
                  borderRadius: '12px',
                  padding: '30px 20px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: previewUrl ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.02)',
                  transition: 'all 0.2s',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <input
                  id="file-input"
                  ref={fileInputRef}
                  type="file"
                  accept="image/png, image/jpeg, image/jpg, image/webp"
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                />

                {previewUrl ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                    <img
                      src={previewUrl}
                      alt="QR Preview"
                      style={{
                        maxWidth: '180px',
                        maxHeight: '180px',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        boxShadow: '0 8px 16px rgba(0,0,0,0.5)',
                        objectFit: 'contain',
                        background: '#ffffff',
                        padding: '4px'
                      }}
                    />
                    <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontWeight: 500 }}>
                      Click or drop another image to replace
                    </span>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(6, 182, 212, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)' }}>
                      <UploadCloud size={24} />
                    </div>
                    <div>
                      <p style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-main)' }}>
                        Drag & Drop QR Image here
                      </p>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                        Supports PNG, JPG, JPEG, or WEBP
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <button
                id="analyze-btn"
                onClick={handleScan}
                disabled={!selectedFile || loading}
                style={{
                  width: '100%',
                  marginTop: '16px',
                  padding: '12px 20px',
                  borderRadius: '10px',
                  border: 'none',
                  background: !selectedFile || loading
                    ? 'rgba(255, 255, 255, 0.08)'
                    : 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
                  color: !selectedFile || loading ? 'var(--text-dim)' : '#ffffff',
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  cursor: !selectedFile || loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  boxShadow: !selectedFile || loading ? 'none' : '0 4px 14px rgba(6, 182, 212, 0.3)',
                  transition: 'all 0.2s',
                }}
              >
                {loading ? (
                  <>
                    <RefreshCw size={18} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
                    Running Multimodal Inference...
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    Analyze QR Code
                  </>
                )}
              </button>

              {/* Test Presets */}
              <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Quick Test Presets
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px', marginTop: '10px' }}>
                  <button
                    id="sample-safe-btn"
                    onClick={() => loadSample('/samples/sample_safe_qr.png', 'sample_safe_qr.png')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '8px 12px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-muted)',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <CheckCircle2 size={14} color="#10b981" />
                    <span>Safe QR (Wikipedia reference)</span>
                  </button>
                  <button
                    id="sample-phish-btn"
                    onClick={() => loadSample('/samples/sample_phishing_qr.png', 'sample_phishing_qr.png')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '8px 12px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-muted)',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <AlertTriangle size={14} color="#f43f5e" />
                    <span>Suspicious QR (Account update token)</span>
                  </button>
                  <button
                    id="sample-nonqr-btn"
                    onClick={() => loadSample('/samples/sample_non_qr.png', 'sample_non_qr.png')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '8px 12px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-muted)',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <XCircle size={14} color="var(--text-dim)" />
                    <span>Non-QR Image (Error test)</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Pipeline Specs Card */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Activity size={16} color="var(--accent-cyan)" />
                Architecture Specifications
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.8rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>URL Branch:</span>
                  <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>Random Forest (55% weight)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>Vision Branch:</span>
                  <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>MobileNetV2 (45% weight)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>Calibration:</span>
                  <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>Isotonic Regression</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>Decision Threshold:</span>
                  <span style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>0.5434</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Scan Results & Explainability */}
          <div>
            {error && (
              <div
                id="error-card"
                className="animate-fade-in"
                style={{
                  padding: '20px',
                  borderRadius: '12px',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  marginBottom: '20px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                }}
              >
                <AlertTriangle size={24} color="#f87171" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <h4 style={{ fontWeight: 600, color: '#fca5a5', marginBottom: '4px' }}>Analysis Notice</h4>
                  <p style={{ fontSize: '0.85rem', color: '#fca5a5' }}>{error}</p>
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div
                className="glass-panel"
                style={{
                  padding: '60px 30px',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '380px',
                }}
              >
                <div
                  style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--text-dim)',
                    marginBottom: '16px',
                  }}
                >
                  <Eye size={30} />
                </div>
                <h3 className="font-display" style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px' }}>
                  Awaiting QR Code Upload
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', maxWidth: '420px', margin: '0 auto' }}>
                  Upload a QR image or select one of the test presets on the left to inspect the decoded destination and multimodal risk score.
                </p>
              </div>
            )}

            {loading && (
              <div
                className="glass-panel pulse-glow"
                style={{
                  padding: '60px 30px',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '380px',
                }}
              >
                <div style={{ width: '48px', height: '48px', marginBottom: '20px', color: 'var(--accent-cyan)' }}>
                  <RefreshCw size={48} style={{ animation: 'spin 1.5s linear infinite' }} />
                </div>
                <h3 className="font-display" style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '8px' }}>
                  Analyzing QR Code Pipeline
                </h3>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <span>1. Local OpenCV QR Matrix Decoding</span>
                  <span>2. 29 Lexical URL Feature Extraction</span>
                  <span>3. MobileNetV2 224x224 Deep Saliency</span>
                  <span>4. Prediction-Level Soft Voting & Isotonic Calibration</span>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                
                {/* 1. Primary Verdict Card */}
                <div
                  id="verdict-card"
                  className="glass-panel"
                  style={{
                    padding: '24px',
                    borderColor: isMalicious ? 'var(--status-malicious-border)' : 'var(--status-benign-border)',
                    background: isMalicious ? 'rgba(244, 63, 94, 0.05)' : 'rgba(16, 185, 129, 0.05)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                      <div
                        style={{
                          width: '52px',
                          height: '52px',
                          borderRadius: '12px',
                          background: isMalicious ? 'var(--status-malicious-bg)' : 'var(--status-benign-bg)',
                          border: `1px solid ${isMalicious ? 'var(--status-malicious-border)' : 'var(--status-benign-border)'}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: isMalicious ? '#f43f5e' : '#10b981',
                        }}
                      >
                        {isMalicious ? <ShieldAlert size={30} /> : <ShieldCheck size={30} />}
                      </div>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span
                            className={isMalicious ? 'badge badge-malicious' : 'badge badge-benign'}
                            style={{ fontSize: '0.9rem', padding: '6px 14px' }}
                          >
                            {result.prediction}
                          </span>
                          <span
                            className="badge badge-neutral"
                            style={{
                              borderColor: isMalicious ? '#f43f5e' : '#10b981',
                              color: isMalicious ? '#fb7185' : '#34d399',
                            }}
                          >
                            {result.risk_level}: {result.risk_label}
                          </span>
                        </div>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                          Calibrated Quishing Probability: <strong>{(result.calibrated_probability * 100).toFixed(2)}%</strong> (Threshold: {(result.threshold * 100).toFixed(1)}%)
                        </p>
                      </div>
                    </div>

                    {/* Score Gauge */}
                    <div style={{ minWidth: '160px', textAlign: 'right' }}>
                      <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'var(--font-display)', color: isMalicious ? '#fb7185' : '#34d399' }}>
                        {riskPercent}%
                      </div>
                      <div className="gauge-track" style={{ marginTop: '4px' }}>
                        <div
                          className="gauge-fill"
                          style={{
                            width: `${riskPercent}%`,
                            background: isMalicious
                              ? 'linear-gradient(90deg, #f59e0b, #f43f5e)'
                              : 'linear-gradient(90deg, #06b6d4, #10b981)',
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. Decoded Inert Destination URL Card */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Decoded Destination URL (Inert String)
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '0.68rem' }}>
                      STATIC ONLY — NEVER VISITED
                    </span>
                  </div>
                  <div
                    style={{
                      background: 'rgba(0, 0, 0, 0.4)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      padding: '12px 14px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '12px',
                    }}
                  >
                    <code
                      id="decoded-url-text"
                      className="font-mono"
                      style={{
                        fontSize: '0.85rem',
                        color: 'var(--accent-cyan)',
                        wordBreak: 'break-all',
                      }}
                    >
                      {result.decoded_url}
                    </code>
                    <button
                      id="copy-url-btn"
                      onClick={() => copyToClipboard(result.decoded_url)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: copied ? '#10b981' : 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                      title="Copy URL string"
                    >
                      {copied ? <Check size={18} /> : <Copy size={18} />}
                    </button>
                  </div>
                </div>

                {/* 3. Branch Probability Breakdown Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  {/* URL Branch */}
                  <div className="glass-panel" style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                        URL Random Forest
                      </span>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Weight: 55%</span>
                    </div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}>
                      {(result.url_probability * 100).toFixed(1)}%
                    </div>
                    <div className="gauge-track" style={{ marginTop: '6px' }}>
                      <div
                        className="gauge-fill"
                        style={{
                          width: `${Math.round(result.url_probability * 100)}%`,
                          background: 'linear-gradient(90deg, #38bdf8, #06b6d4)',
                        }}
                      />
                    </div>
                  </div>

                  {/* Image Branch */}
                  <div className="glass-panel" style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                        MobileNetV2 Vision
                      </span>
                      <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Weight: 45%</span>
                    </div>
                    <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}>
                      {(result.image_probability * 100).toFixed(1)}%
                    </div>
                    <div className="gauge-track" style={{ marginTop: '6px' }}>
                      <div
                        className="gauge-fill"
                        style={{
                          width: `${Math.round(result.image_probability * 100)}%`,
                          background: 'linear-gradient(90deg, #818cf8, #6366f1)',
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* 4. Explainable Evidence Tabs */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '16px' }}>
                    <button
                      id="tab-url-btn"
                      onClick={() => setActiveTab('url')}
                      style={{
                        background: activeTab === 'url' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                        border: `1px solid ${activeTab === 'url' ? 'var(--accent-cyan)' : 'transparent'}`,
                        color: activeTab === 'url' ? 'var(--accent-cyan)' : 'var(--text-muted)',
                        padding: '6px 14px',
                        borderRadius: '8px',
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                      }}
                    >
                      <Layers size={16} />
                      URL Lexical Evidence ({result.explanation?.url?.length || 0})
                    </button>
                    <button
                      id="tab-vision-btn"
                      onClick={() => setActiveTab('vision')}
                      style={{
                        background: activeTab === 'vision' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                        border: `1px solid ${activeTab === 'vision' ? 'var(--accent-cyan)' : 'transparent'}`,
                        color: activeTab === 'vision' ? 'var(--accent-cyan)' : 'var(--text-muted)',
                        padding: '6px 14px',
                        borderRadius: '8px',
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                      }}
                    >
                      <ImageIcon size={16} />
                      Visual Saliency (Grad-CAM)
                    </button>
                  </div>

                  {/* Tab 1: URL Evidence */}
                  {activeTab === 'url' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {result.explanation?.url?.map((item, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: '12px 14px',
                            background: 'rgba(255, 255, 255, 0.02)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '8px',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '4px',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span
                                className="badge"
                                style={{
                                  fontSize: '0.68rem',
                                  padding: '2px 8px',
                                  background: item.impact === 'elevated_risk' ? 'var(--status-malicious-bg)' : 'var(--status-benign-bg)',
                                  color: item.impact === 'elevated_risk' ? '#fb7185' : '#34d399',
                                  border: `1px solid ${item.impact === 'elevated_risk' ? 'var(--status-malicious-border)' : 'var(--status-benign-border)'}`,
                                }}
                              >
                                {item.impact === 'elevated_risk' ? 'Elevated Risk Signal' : 'Mitigating Factor'}
                              </span>
                              <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-main)' }}>
                                {item.feature.replace(/_/g, ' ')}
                              </span>
                            </div>
                            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                              Value: {item.value}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                            {item.finding}
                          </p>
                          {item.description && (
                            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                              {item.description}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Tab 2: Visual Grad-CAM */}
                  {activeTab === 'vision' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {result.explanation?.gradcam?.available ? (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'center' }}>
                          <div style={{ textAlign: 'center' }}>
                            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', display: 'block', marginBottom: '8px' }}>
                              Original Preprocessed (224x224)
                            </span>
                            <img
                              src={previewUrl}
                              alt="Original QR"
                              style={{
                                width: '100%',
                                maxWidth: '224px',
                                borderRadius: '8px',
                                border: '1px solid var(--border-color)',
                                background: '#ffffff',
                                padding: '4px'
                              }}
                            />
                          </div>
                          <div style={{ textAlign: 'center' }}>
                            <span style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)', display: 'block', marginBottom: '8px', fontWeight: 600 }}>
                              Grad-CAM Saliency Overlay
                            </span>
                            <img
                              id="gradcam-image"
                              src={result.explanation.gradcam.data_url}
                              alt="Grad-CAM Overlay"
                              style={{
                                width: '100%',
                                maxWidth: '224px',
                                borderRadius: '8px',
                                border: '1px solid var(--accent-cyan)',
                                boxShadow: '0 4px 20px rgba(6, 182, 212, 0.25)',
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <p style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
                          Grad-CAM heatmap could not be generated for this image.
                        </p>
                      )}

                      {/* Visual Findings List */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
                        {result.explanation?.image?.map((item, idx) => (
                          <div
                            key={idx}
                            style={{
                              padding: '10px 12px',
                              background: 'rgba(255, 255, 255, 0.02)',
                              border: '1px solid var(--border-color)',
                              borderRadius: '8px',
                            }}
                          >
                            <span style={{ fontWeight: 600, fontSize: '0.82rem', color: 'var(--accent-cyan)' }}>
                              {item.component}:
                            </span>
                            <p style={{ fontSize: '0.82rem', color: 'var(--text-main)', marginTop: '2px' }}>
                              {item.finding}
                            </p>
                            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                              {item.detail}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid var(--border-color)',
          background: 'rgba(10, 13, 20, 0.95)',
          padding: '20px 24px',
          fontSize: '0.78rem',
          color: 'var(--text-dim)',
          textAlign: 'center',
        }}
      >
        <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div>
            <strong>QRShield</strong> — CSET485 AI and Society Course Project
          </div>
          <div>
            Team: Harshit Pahuja (s24cseu0736) • Hemang Nag (s24cseu0728) • Saatvik Sharma (s24cseu2097) • Tanish (s24cseu0589) • Rohit Upadhyay (s24cseu1359)
          </div>
          <div style={{ marginTop: '4px', fontSize: '0.72rem' }}>
            Disclaimer: Metrics reported (Fusion Test Accuracy 0.988, ROC-AUC 0.997, Brier 0.010) reflect the internal 10,000-sample development split. Real-world distribution shifts may alter detection efficacy.
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
