import React, { useEffect, useState } from "react";
import "../App.css";
import Inputbox from "./Inputbox";
import Cards from "./Cards";
import MainDash from "./MainDash";
import Loader from "./Loader";

function RightSide({ dataset }) {
	console.log(dataset);
	const [first, setfirst] = useState({
		twitter: {
			overall_results: {
				overall_prediction: 0,
				overall_sentiment: "-",
			},
			best_model: "----",
			results: [
				{
					prediction: "----",
					confidence: 100,
				},
				{
					prediction: "----",
					confidence: 100,
				},
				{
					prediction: "----",
					confidence: 100,
				},
			],
		},
		imdb: {
			overall_results: {
				overall_prediction: 0,
				overall_sentiment: "-",
			},
			best_model: "----",
			results: [
				{
					prediction: "----",
					confidence: 100,
				},
				{
					prediction: "----",
					confidence: 100,
				},
				{
					prediction: "----",
					confidence: 100,
				},
			],
		},
	});
	const [isLoading, setIsLoading] = useState(false);
	return (
		<div className="Rightside">
			<div className="first">
				<Inputbox
					first={first}
					setfirst={setfirst}
					setIsLoading={setIsLoading}
				/>
			</div>
			{isLoading ? (
				<Loader />
			) : (
				<>
					<div className="second">
						<Cards first={first} setfirst={setfirst} dataset={dataset} />
					</div>
					<div className="third">
						<MainDash first={first} setfirst={setfirst} dataset={dataset} />
					</div>
				</>
			)}
		</div>
	);
}

export default RightSide;
