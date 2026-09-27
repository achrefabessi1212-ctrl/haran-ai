import { useState } from "react";
import BarnGuardian from "./components/BarnGuardian.jsx";
import ElderHeader from "./components/ElderHeader.jsx";
import FarmBook from "./components/FarmBook.jsx";
import FieldGuardian from "./components/FieldGuardian.jsx";

export default function App() {
  const [activeTab, setActiveTab] = useState("field");

  return (
    <div className="app-shell">
      <ElderHeader activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="app-main">
        {activeTab === "field" && <FieldGuardian />}
        {activeTab === "barn" && <BarnGuardian />}
        {activeTab === "book" && <FarmBook />}
      </main>
    </div>
  );
}
