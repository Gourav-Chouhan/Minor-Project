import React from "react";
import "../App.css";
import { useState } from "react";
import { IconButton } from "rsuite";
import SearchIcon from "@mui/icons-material/Search";
import axios from "axios";

function Inputbox({ first, setfirst, setIsLoading }) {
	const [data, setdata] = useState(null);
	function getData(val) {
		setdata(val.target.value);
	}
	const handleClick = (event) => {
		setIsLoading(true);
		axios({
			url: "http://localhost:5000/predict_all",
			method: "POST",
			data: { text: data },
			headers: { "Content-Type": "application/json" },
		})
			.then((response) => {
				setfirst(response.data);
				setIsLoading(false);
			})
			.catch((error) => {
				console.log(error);
				setIsLoading(false);
			});

		console.log({ data });
	};
	return (
		<div className="inputbox">
			<input
				type="text"
				className="input"
				placeholder="Enter your sentence here"
				onChange={getData}
				onKeyUp={(e) => {
					if (e.key === "Enter") {
						handleClick();
					}
				}}
			/>
			<IconButton
				className="search"
				icon={<SearchIcon />}
				onClick={handleClick}
			></IconButton>
		</div>
	);
}

export default Inputbox;
