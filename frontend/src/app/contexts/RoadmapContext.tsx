import { createContext ,useContext,useState,ReactNode} from "react";

interface RoadmapData {
  roadmap_id: string;
  title: string;
  topic: string;
  description: string;
  total_nodes: number;
}

interface RoadmapContextType{
  currentRoadmap:RoadmapData|null,
  setRoadmap:(roadmap:RoadmapData)=>void;
  clearRoadmap:()=>void;
}

const RoadmapContext=createContext<RoadmapContextType|undefined>(undefined);

export function RoadmapProvider({children}:{children:ReactNode}){
    const [currentRoadmap,setcurrentRoadmap]=useState<RoadmapData|null>(() => {
        const saved=localStorage.getItem("active_roadmap")
        return saved?JSON.parse(saved):null;
    });
    const setRoadmap = (notebook: RoadmapData) => {
    setcurrentRoadmap(notebook);
    localStorage.setItem("active_roadmap", JSON.stringify(notebook));
  };

  const clearRoadmap = () => {
    setcurrentRoadmap(null);
    localStorage.removeItem("active_roadmap");
  };

  return (
    <RoadmapContext.Provider value={{ currentRoadmap, setRoadmap, clearRoadmap }}>
      {children}
    </RoadmapContext.Provider>
  );
}

export function useRoadmap() {
  const context = useContext(RoadmapContext);
  if (context === undefined) {
    throw new Error("useRoadmap must be used within a RoadmapProvider");
  }
  return context;
}
