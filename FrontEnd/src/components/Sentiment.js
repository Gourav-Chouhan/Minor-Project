import React from "react";
import "../App.css";

function Sentiment(props) {
	const mystyle = {
		color: "green",
	};
	return (
		<div className="Sentiment">
			<h3>Overall Sentiment Analysis Result </h3>
			<div className="analysis">
				<span className="result" style={mystyle}>
					{props.result}
				</span>
				<span className="score">Confidence : {props.confidence}</span>
				<span className="score">Best Result : {props.model}</span>
			</div>
		</div>
	);
}

export default Sentiment;
