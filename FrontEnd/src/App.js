import "./App.css";
import Sidebar from "./components/Sidebar";
import RightSide from "./components/RightSide";
import Dashboard from "./components/Dashboard";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
	return (
		<div className="App">
			<BrowserRouter>
				<Sidebar />
				<Routes>
					<Route path="/" element={<Dashboard />} />
					<Route path="/dashboard" element={<Dashboard />} />
					<Route path="/twitter" element={<RightSide dataset={"twitter"} />} />
					<Route path="/imdb" element={<RightSide dataset={"imdb"} />} />
				</Routes>
			</BrowserRouter>
		</div>
	);
}

export default App;
