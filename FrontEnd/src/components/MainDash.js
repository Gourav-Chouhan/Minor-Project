import React from "react";
import "../App.css";
import Graph from "./Graph";
import Sentiment from "./Sentiment";

function MainDash(props) {
	const models = ["Simple ANN", "CNN", "Bi-LSTM"];
	return (
		<div className="Maindash">
			{/* {console.log(props.first['overall_results'])} */}
			<div className="one">
				<Sentiment
					result={props.first[props.dataset]["overall_prediction_sentiment"]}
					confidence={props.first[props.dataset]["overall_prediction"]}
					model={models[props.first[props.dataset]["best_model"]]}
				/>
			</div>
			<div className="two">
				<Graph />
			</div>
		</div>
	);
}

export default MainDash;
