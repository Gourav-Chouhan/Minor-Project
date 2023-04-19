import React from "react";
import "../App.css";
import { Carddata } from "./Carddata";
import Card from "./Card";

function Cards({ first, setfirst, dataset }) {
	const colors = [
		{
			backGround: "rgba(85, 176, 241, 1)",
			boxShadow: "0px 10px 20px 0px rgba(85, 176, 241, 0.6)",
		},
		{
			backGround: "rgba(246, 176, 40, 1)",
			boxShadow: "0px 10px 20px 0px rgba(246, 176, 40, 0.6)",
		},
		{
			backGround: "rgba(20, 151, 119, 1)",
			boxShadow: "0px 10px 20px 0px rgba(20, 151, 119, 0.6)",
		},
	];
	return (
		<div className="Cards">
			{/* {Carddata.map((card,id)=>{ */}
			{/* return( */}
			<>
				<div className="parentcontainer">
					<Card
						first={first}
						setfirst={setfirst}
						title="Simple Neural Network"
						color={colors[0]}
						barValue={first[dataset]["results"][0]["confidence"]}
						prediction={first[dataset]["results"][0]["prediction"]}
					/>
				</div>
				<div className="parentcontainer">
					<Card
						first={first}
						setfirst={setfirst}
						title="CNN"
						color={colors[1]}
						prediction={first[dataset]["results"][1]["prediction"]}
						barValue={first[dataset]["results"][1]["confidence"]}
					/>
				</div>
				<div className="parentcontainer">
					<Card
						first={first}
						setfirst={setfirst}
						title="BiLSTM"
						color={colors[2]}
						prediction={first[dataset]["results"][2]["prediction"]}
						barValue={first[dataset]["results"][2]["confidence"]}
					/>
				</div>
			</>
			{/* ) */}
			{/* })} */}
		</div>
	);
}

export default Cards;
