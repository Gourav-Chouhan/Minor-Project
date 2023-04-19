import React from "react";
import "../App.css";
import { Sidebardata } from "./Sidebardata";
import { useNavigate } from "react-router-dom";

function Sidebar() {
	const navigate = useNavigate();
	return (
		<div className="Sidebar">
			<ul className="SidebarList">
				{Sidebardata.map((val, key) => {
					return (
						<li
							className="row"
							id={window.location.pathname == val.link ? "active" : ""}
							key={key}
							onClick={() => {
								// window.location.pathname = val.link;
								navigate(val.link)
								// <Navigate to={val.link} replace={true} />
							}}
						>
							<div id="icon">{val.icon}</div> <div id="title">{val.title}</div>
						</li>
					);
				})}
			</ul>
		</div>
	);
}

export default Sidebar;
