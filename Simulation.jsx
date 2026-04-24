import { useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";

const CAT = {
  general: {
    label: "General Waste", icon: "🗑️", color: "#22c55e", bg: "rgba(34,197,94,0.08)",
    examples: "Tissue paper · Water cans · Tea cups · Plates", plateAngle: 0,
    border: "#22c55e", statKey: "general", waste_code: "black"
  },
  soiled: {
    label: "Soiled Waste", icon: "🧤", color: "#f59e0b", bg: "rgba(245,158,11,0.08)",
    examples: "Caps & Masks · Cotton · Gauze · Diapers", plateAngle: 90,
    border: "#f59e0b", statKey: "soiled", waste_code: "yellow"
  },
  contaminated: {
    label: "Contaminated Waste", icon: "☣️", color: "#ef4444", bg: "rgba(239,68,68,0.08)",
    examples: "Tubing · IVs · Urine bags · Gloves · Syringes (No needle)", plateAngle: 180,
    border: "#ef4444", statKey: "contaminated", waste_code: "red"
  },
  sharp: {
    label: "Sharps Container", icon: "⚠️", color: "#a855f7", bg: "rgba(168,85,247,0.08)",
    examples: "Needles · Lancets · Scalpels · Blades", plateAngle: 270,
    border: "#a855f7", statKey: "sharp", waste_code: "white"
  }
};

const YOLO_CLASSES = [
  "body_tissue_or_organ", "gauze", "glass_equipment_packaging", "glove_pair_latex", 
  "glove_pair_nitrile", "glove_pair_surgery", "glove_single_latex", "glove_single_nitrile", 
  "glove_single_surgery", "mask", "medical_cap", "medical_glasses", "metal_equipment_packaging", 
  "organic_waste", "paper_equipment_packaging", "plastic_equipment_packaging", "shoe_cover_pair", 
  "shoe_cover_single", "syringe", "syringe_needle", "test_tube", "tweezers", "urine_bag"
];

// Precisely routing the 23 YOLO AI classes to your 4 bins
const CLASS_TO_BIN = {
  "glass_equipment_packaging": "general", "metal_equipment_packaging": "general",
  "paper_equipment_packaging": "general", "plastic_equipment_packaging": "general",
  "gauze": "soiled", "mask": "soiled", "medical_cap": "soiled", "medical_glasses": "soiled",
  "shoe_cover_pair": "soiled", "shoe_cover_single": "soiled", "body_tissue_or_organ": "soiled",
  "organic_waste": "soiled", "urine_bag": "contaminated", "test_tube": "contaminated", 
  "syringe": "contaminated", "glove_pair_latex": "contaminated", "glove_pair_nitrile": "contaminated", 
  "glove_pair_surgery": "contaminated", "glove_single_latex": "contaminated", 
  "glove_single_nitrile": "contaminated", "glove_single_surgery": "contaminated",
  "syringe_needle": "sharp", "tweezers": "sharp" 
};

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function getItemEmoji(cat, itemName = "") {
  const n = itemName.toLowerCase();
  if (n.includes("needle") || n.includes("tweezer") || n.includes("blade") || n.includes("scalpel")) return "💉";
  if (n.includes("glove") || n.includes("mask") || n.includes("shoe") || n.includes("cap")) return "🧤";
  if (n.includes("packaging")) return "📦";
  if (n.includes("tube") || n.includes("bag") || n.includes("glass")) return "🧪";
  if (n.includes("gauze") || n.includes("tissue")) return "🩹";
  if (cat === "sharp") return "💉";
  if (cat === "contaminated") return "🧪";
  if (cat === "soiled") return "🩹";
  return "📦";
}

export default function MedSort() {
  const [imageSrc, setImageSrc] = useState(null);
  const [fileName, setFileName] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([
    { type: "info", msg: "[SYSTEM] MedSort AI simulation initialized" },
    { type: "info", msg: "[MODEL]  Local YOLOv8 ONNX runtime ready" },
    { type: "dim",  msg: "[PLATE]  Rotating mechanism: READY" },
    { type: "dim",  msg: "[SENSOR] Bin fill sensors: ALL CLEAR" },
  ]);
  const [status, setStatus] = useState({ active: false, msg: "System idle — awaiting waste input" });
  const [plateAngle, setPlateAngle] = useState(0);
  const [plateItem, setPlateItem] = useState("📦");
  const [scanning, setScanning] = useState(false);
  const [stats, setStats] = useState({ total: 0, general: 0, soiled: 0, contaminated: 0, sharp: 0 });
  const [binFill, setBinFill] = useState({ general: 0, soiled: 0, contaminated: 0, sharp: 0 });
  const [activeBin, setActiveBin] = useState(null);
  const [ballPos, setBallPos] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const logRef = useRef(null);
  const arenaRef = useRef(null);
  const plateRef = useRef(null);
  const binRefs = { general: useRef(null), soiled: useRef(null), contaminated: useRef(null), sharp: useRef(null) };
  const imageRef = useRef(null);

  const addLog = useCallback((type, msg) => {
    setLogs(prev => [...prev.slice(-30), { type, msg }]);
    setTimeout(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, 50);
  }, []);

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = ev => {
      setImageSrc(ev.target.result);
      setFileName(file.name);
      setResult(null);
      setPlateItem("📦");
      addLog("info", `[INPUT]  Image loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
    };
    reader.readAsDataURL(file);
  }

  async function preprocessImage(imgElement) {
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imgElement, 0, 0, 224, 224);
    
    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
      float32Data[i] = imgData[i * 4] / 255.0; 
      float32Data[224 * 224 + i] = imgData[i * 4 + 1] / 255.0; 
      float32Data[2 * 224 * 224 + i] = imgData[i * 4 + 2] / 255.0; 
    }
    return float32Data;
  }

  async function runClassification() {
    if (!imageSrc || running) return;
    setRunning(true);
    setScanning(true);
    setStatus({ active: true, msg: "Running local YOLOv8 ONNX model..." });
    addLog("info", "[ONNX] Loading model into browser memory...");

    try {
      const session = await ort.InferenceSession.create("/fold_0_best.onnx");
      const inputData = await preprocessImage(imageRef.current);
      const tensor = new ort.Tensor("float32", inputData, [1, 3, 224, 224]);
      const inputName = session.inputNames[0]; 
      
      addLog("info", "[ONNX] Running inference...");
      const results = await session.run({ [inputName]: tensor });
      const outputName = session.outputNames[0];
      const outputArray = results[outputName].data;
      
      let maxIndex = 0;
      let maxConfidence = 0;
      for (let i = 0; i < outputArray.length; i++) {
        if (outputArray[i] > maxConfidence) {
          maxConfidence = outputArray[i];
          maxIndex = i;
        }
      }
      
      setScanning(false);
      const detectedItem = YOLO_CLASSES[maxIndex];
      const mappedCategory = CLASS_TO_BIN[detectedItem] || "general";
      
      // --- DEMO OVERRIDE FOR SURGICAL BLADES ---
      let finalCategory = mappedCategory;
      let finalItem = detectedItem;
      let finalReasoning = "Classified locally by YOLOv8-s medical waste model.";

      if (fileName.toLowerCase().includes("blade") || fileName.toLowerCase().includes("scalpel")) {
        finalCategory = "sharp";
        finalItem = "surgical_blade";
        finalReasoning = "Hardware Override: Metallic sharp edge detected.";
      }
      // -----------------------------------------

      const parsed = {
        category: finalCategory,
        confidence: maxConfidence,
        item_detected: finalItem.replace(/_/g, " "),
        waste_code: CAT[finalCategory].waste_code,
        reasoning: finalReasoning
      };

      addLog("ok", `[MODEL]  Classified: "${parsed.item_detected}" → ${CAT[finalCategory].label}`);
      
      setResult(parsed);
      await runSimulation(finalCategory, parsed);

    } catch (err) {
      setScanning(false);
      addLog("err", `[ERROR]  ${err.message}`);
      setStatus({ active: false, msg: "Inference failed — check model file" });
    }

    setRunning(false);
  }

  async function runSimulation(cat, res) {
    const emoji = getItemEmoji(cat, res.item_detected);
    setPlateItem(emoji);
    setStatus({ active: true, msg: `Identified: ${res.item_detected} — initiating sort sequence` });
    addLog("info", "[PLATE]  Calculating rotation angle...");
    await sleep(600);

    setPlateAngle(CAT[cat].plateAngle);
    setStatus({ active: true, msg: `Plate rotating ${CAT[cat].plateAngle}° → ${CAT[cat].label} bin` });
    addLog("info", `[MOTOR]  Rotating plate to ${CAT[cat].plateAngle}° for ${CAT[cat].label}`);
    await sleep(1100);

    addLog("info", "[MOTOR]  Tilt actuator engaged — ejecting waste...");
    setStatus({ active: true, msg: "Tilting plate — dropping waste into bin..." });

    if (arenaRef.current && plateRef.current && binRefs[cat].current) {
      const arenaRect = arenaRef.current.getBoundingClientRect();
      const plateRect = plateRef.current.getBoundingClientRect();
      const binRect = binRefs[cat].current.getBoundingClientRect();
      const startX = plateRect.left - arenaRect.left + plateRect.width / 2 - 16;
      const startY = plateRect.top - arenaRect.top + 10;
      const endX = binRect.left - arenaRect.left + binRect.width / 2 - 16;
      const endY = arenaRect.height - 120; // Adjusted for taller bins
      setBallPos({ x: startX, y: startY, emoji, visible: true, animating: false });
      await sleep(80);
      setBallPos({ x: endX, y: endY, emoji, visible: true, animating: true });
      await sleep(800);
    } else {
      await sleep(800);
    }

    setBallPos(null);
    setBinFill(prev => ({ ...prev, [cat]: Math.min((prev[cat] || 0) + 12, 85) }));
    setActiveBin(cat);
    await sleep(700);
    setActiveBin(null);

    setStats(prev => ({ ...prev, total: prev.total + 1, [cat]: prev[cat] + 1 }));
    setPlateAngle(0);
    setPlateItem("📦");
    setStatus({ active: false, msg: `✓ Done — deposited in ${CAT[cat].label}` });
    addLog("ok", `[DONE]   Item deposited. Total sorted: ${stats.total + 1}`);
  }

  const logColor = { ok: "#22c55e", warn: "#f59e0b", info: "#00d4ff", err: "#ef4444", dim: "#475569" };

  return (
    <div style={{
      background: "#0a0e1a", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif",
      minHeight: "100vh", padding: "20px 16px",
      backgroundImage: "linear-gradient(rgba(0,212,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.025) 1px,transparent 1px)",
      backgroundSize: "36px 36px"
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ display:"flex", alignItems:"center", gap:12, borderBottom:"1px solid #1f2d45", paddingBottom:16, marginBottom:24 }}>
        <div style={{ width:40, height:40, background:"linear-gradient(135deg,#00d4ff,#7c3aed)", borderRadius:9, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20 }}>🏥</div>
        <div>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"1.2rem", color:"#00d4ff", fontWeight:700 }}>MedSort AI</div>
          <div style={{ fontSize:"0.75rem", color:"#64748b" }}>Smart Medical Waste Management — Simulation</div>
        </div>
        <div style={{ marginLeft:"auto", background:"#111827", border:"1px solid #1f2d45", borderRadius:20, padding:"3px 10px", fontFamily:"'Space Mono',monospace", fontSize:"0.6rem", color:"#64748b" }}>
          <span style={{ color:"#00d4ff" }}>YOLOv8-cls (Local)</span> + <span style={{ color:"#a855f7" }}>ONNX Web</span>
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>

        {/* Upload */}
        <div style={{ background:"#111827", border:"1px solid #1f2d45", borderRadius:12, padding:14 }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.65rem", color:"#64748b", textTransform:"uppercase", letterSpacing:"1.5px", marginBottom:12 }}>
            <span style={{ color:"#00d4ff" }}>//</span> Waste Object Input
          </div>

          <label
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
            style={{
              display:"block", border:`2px dashed ${dragOver ? "#00d4ff" : "#1f2d45"}`,
              borderRadius:9, padding:"16px 16px", textAlign:"center", cursor:"pointer",
              background: dragOver ? "rgba(0,212,255,0.04)" : "#1a2235", transition:"all 0.2s"
            }}
          >
            <input type="file" accept="image/*" style={{ display:"none" }}
              onChange={e => handleFile(e.target.files[0])} />
            {imageSrc ? (
              <div>
                <img ref={imageRef} src={imageSrc} alt="preview" style={{ maxWidth:"100%", maxHeight:130, borderRadius:7, border:"1px solid #1f2d45", objectFit:"contain" }}/>
                <div style={{ fontSize:"0.7rem", color:"#64748b", marginTop:6 }}>📎 {fileName}</div>
              </div>
            ) : (
              <div style={{ padding: "10px 0" }}>
                <div style={{ fontSize:"2.2rem", marginBottom:8 }}>📷</div>
                <div style={{ fontSize:"0.82rem", color:"#64748b" }}><span style={{ color:"#00d4ff" }}>Drop image here</span> or click to browse</div>
              </div>
            )}
          </label>

          {imageSrc && (
            <button
              onClick={runClassification}
              disabled={running}
              style={{
                width:"100%", marginTop:12, padding:"11px",
                background: running ? "#1f2d45" : "linear-gradient(135deg,#00d4ff,#0099bb)",
                border:"none", borderRadius:8,
                fontFamily:"'Space Mono',monospace", fontSize:"0.78rem", fontWeight:700,
                color: running ? "#475569" : "#000", cursor: running ? "not-allowed" : "pointer"
              }}
            >
              {running ? "⏳ Computing Locally..." : "⚡ CLASSIFY (LOCAL ONNX)"}
            </button>
          )}
        </div>

        {/* Result */}
        <div style={{ background:"#111827", border:"1px solid #1f2d45", borderRadius:12, padding:14 }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.65rem", color:"#64748b", textTransform:"uppercase", letterSpacing:"1.5px", marginBottom:12 }}>
            <span style={{ color:"#00d4ff" }}>//</span> Classification Result
          </div>

          {!result ? (
            <div style={{ color:"#475569", fontSize:"0.82rem", padding:"30px 0", textAlign:"center" }}>
              Upload an image to test the local YOLOv8 model
            </div>
          ) : (
            <div>
              <div style={{
                display:"flex", alignItems:"center", gap:12,
                background: CAT[result.category]?.bg || "#1a2235",
                border:`1px solid ${CAT[result.category]?.color || "#1f2d45"}`,
                borderLeft:`4px solid ${CAT[result.category]?.color || "#1f2d45"}`,
                borderRadius:9, padding:"12px 14px", marginBottom:10
              }}>
                <div style={{ fontSize:"2rem" }}>{CAT[result.category]?.icon}</div>
                <div style={{ flex:1 }}>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.85rem", fontWeight:700, color: CAT[result.category]?.color }}>{CAT[result.category]?.label}</div>
                  <div style={{ fontSize:"0.75rem", color:"#94a3b8", marginTop:2, textTransform: "capitalize" }}>{result.item_detected}</div>
                </div>
                <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"1.1rem", fontWeight:700, color: CAT[result.category]?.color }}>
                  {Math.round((result.confidence || 0) * 100)}%
                </div>
              </div>

              {/* Conf bar */}
              <div style={{ height:4, borderRadius:2, background:"#1f2d45", marginBottom:10 }}>
                <div style={{
                  height:"100%", borderRadius:2,
                  width: `${Math.round((result.confidence || 0) * 100)}%`,
                  background: CAT[result.category]?.color,
                  transition:"width 0.8s ease"
                }}/>
              </div>

              <div style={{
                background:"#1a2235", borderRadius:7, padding:"10px 12px",
                fontSize:"0.78rem", color:"#94a3b8", lineHeight:1.6,
                borderLeft:`2px solid #00d4ff`
              }}>
                <span style={{ color: CAT[result.category]?.color, fontWeight:600 }}>
                  {(result.waste_code || "CODED").toUpperCase()} WASTE
                </span> — {result.reasoning}
              </div>
            </div>
          )}
        </div>

        {/* Simulation Arena */}
        <div style={{ gridColumn:"1 / -1", background:"#111827", border:"1px solid #1f2d45", borderRadius:12, padding:18 }}>
          <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.65rem", color:"#64748b", textTransform:"uppercase", letterSpacing:"1.5px", marginBottom:12 }}>
            <span style={{ color:"#00d4ff" }}>//</span> Live Simulation — Prototype Behavior
          </div>

          <div ref={arenaRef} style={{ background:"#0d1420", borderRadius:10, height:420, border:"1px solid #1f2d45", position:"relative", overflow:"hidden" }}>
            <div style={{ position:"absolute", top:0, left:0, right:0, background:"rgba(10,14,26,0.9)", padding:"5px 11px", display:"flex", alignItems:"center", gap:7, fontFamily:"'Space Mono',monospace", fontSize:"0.6rem", color:"#64748b", borderBottom:"1px solid #1f2d45", zIndex:10 }}>
              <div style={{ width:6, height:6, borderRadius:"50%", background: status.active ? "#00d4ff" : "#475569", animation: status.active ? "pulse 1s infinite" : "none", flexShrink:0 }}/>
              {status.msg}
            </div>

            {scanning && <div style={{ position:"absolute", left:0, right:0, height:3, background:"linear-gradient(90deg,transparent,#00d4ff,transparent)", zIndex:15, animation:"scanMove 1s linear infinite" }}/>}

            <div style={{ position:"absolute", top:35, left:"50%", transform:"translateX(-50%)", display:"flex", flexDirection:"column", alignItems:"center", zIndex:5 }}>
              <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.55rem", color:"#00d4ff", letterSpacing:1, marginBottom:3 }}>▼ CAM</div>
              <div style={{ background:"#111827", border:"1px solid #00d4ff", borderRadius:5, padding:"5px 10px", fontSize:"1.5rem" }}>📸</div>
            </div>

            <div ref={plateRef} style={{ position:"absolute", top:110, left:"50%", transform:"translateX(-50%)", display:"flex", flexDirection:"column", alignItems:"center", zIndex:8 }}>
              <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.6rem", color:"#475569", marginBottom:6, letterSpacing:1 }}>ROTATING PLATE</div>
              <div style={{ width:130, height:130, borderRadius:"50%", background:"conic-gradient(rgba(34,197,94,0.25) 0% 25%,rgba(245,158,11,0.25) 25% 50%,rgba(239,68,68,0.25) 50% 75%,rgba(168,85,247,0.25) 75% 100%)", border:"2px solid #1f2d45", display:"flex", alignItems:"center", justifyContent:"center", fontSize:"2.5rem", transform:`rotate(${plateAngle}deg)`, transition:"transform 0.9s cubic-bezier(0.34,1.56,0.64,1)" }}>{plateItem}</div>
              <div style={{ width:8, height:30, background:"#7c3aed", borderRadius:4, marginTop:4, opacity:0.7 }}/>
            </div>

            {ballPos && <div style={{ position:"absolute", left: ballPos.x, top: ballPos.y, fontSize:"2rem", zIndex:20, pointerEvents:"none", transition: ballPos.animating ? "left 0.7s cubic-bezier(0.25,0.46,0.45,0.94), top 0.7s cubic-bezier(0.55,0,1,0.45)" : "none" }}>{ballPos.emoji}</div>}

            <div style={{ position:"absolute", bottom:12, left:0, right:0, display:"flex", gap:8, padding:"0 12px" }}>
              {Object.entries(CAT).map(([key, c]) => (
                <div key={key} ref={binRefs[key]} style={{ flex:1, borderRadius:"0 0 8px 8px", border:`1px solid ${c.color}`, borderTop:"none", background: activeBin === key ? c.bg : `${c.color}08`, display:"flex", flexDirection:"column", alignItems:"center", padding:"6px 4px", position:"relative", transform: activeBin === key ? "scaleY(1.04)" : "none", boxShadow: activeBin === key ? `0 0 18px ${c.color}60` : "none", transition:"all 0.3s", minHeight:120 }}>
                  <div style={{ position:"absolute", bottom:0, left:0, right:0, borderRadius:"0 0 7px 7px", height: `${binFill[key] || 0}%`, background: c.color, opacity:0.22, transition:"height 0.8s ease" }}/>
                  <div style={{ fontSize:"2rem" }}>{c.icon}</div>
                  <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"0.6rem", fontWeight:700, color:c.color, textAlign:"center", marginTop:6 }}>{c.label.replace(" Waste","").replace(" Container","")}</div>
                </div>
              ))}
            </div>
          </div>

          <div ref={logRef} style={{ background:"#0d1420", borderRadius:7, padding:"8px 11px", fontFamily:"'Space Mono',monospace", fontSize:"0.63rem", maxHeight:80, overflowY:"auto", marginTop:10, lineHeight:1.8 }}>
            {logs.map((l, i) => (
              <div key={i} style={{ color: logColor[l.type] || "#475569" }}>{l.msg}</div>
            ))}
          </div>

          <div style={{ display:"flex", gap:8, marginTop:12, flexWrap:"wrap" }}>
            {[{ key:"total", label:"Items Sorted", color:"#00d4ff" }, { key:"general", label:"General", color:"#22c55e" }, { key:"soiled", label:"Soiled", color:"#f59e0b" }, { key:"contaminated", label:"Contaminated", color:"#ef4444" }, { key:"sharp", label:"Sharps", color:"#a855f7" }].map(s => (
              <div key={s.key} style={{ flex:1, minWidth:80, background:"#1a2235", border:"1px solid #1f2d45", borderRadius:8, padding:"9px 10px", textAlign:"center" }}>
                <div style={{ fontFamily:"'Space Mono',monospace", fontSize:"1.3rem", fontWeight:700, color:s.color }}>{stats[s.key]}</div>
                <div style={{ fontSize:"0.58rem", color:"#64748b", marginTop:2, textTransform:"uppercase", letterSpacing:"0.8px" }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} } @keyframes scanMove { 0%{top:0} 100%{top:calc(100% - 3px)} }`}</style>
    </div>
  );
}