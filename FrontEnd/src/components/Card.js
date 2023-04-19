import React from 'react'
import '../App.css'
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function Card(props) {
  const mystyle={
    backgroundColor:props.color.backGround,
    boxShadow:props.color.boxShadow
  };
  // const mystyle={
  //   backgroundColor:"rgba(255, 126, 0, 1)",
  //   boxShadow: "0px 10px 20px 0px rgba(85, 176, 241, 0.6)"
  // };
  return (
    <div className="Compactcard" style={mystyle}>
    <div className="radialBar">
        <CircularProgressbar
          value={props.barValue}
          text={`${props.barValue}%`}
        />
        <div className="card-detail">
        <span className='cardtitle'>{props.title}</span>
        <span className='predict'>Prediction : {props.prediction}</span>
        </div>
    </div>
    </div>
  )
}

export default Card