import "./styles.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./home.jsx";
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" exact component={Home} />
      </Routes>
    </Router>
  );
}
