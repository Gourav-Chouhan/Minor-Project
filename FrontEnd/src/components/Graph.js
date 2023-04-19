import React from 'react'
import {useState} from 'react';
import { Line } from 'react-chartjs-2';
import {Chart as ChartJS, Title, Tooltip, LineElement, Legend, CategoryScale, LinearScale, PointElement, Filler} from 'chart.js';
ChartJS.register(
  Title, Tooltip, LineElement, Legend,
  CategoryScale, LinearScale, PointElement, Filler
)

function Graph() {
    const [data, setData]= useState({
        labels:[0,1,2,3,4,5,6,7],
        scales: {
            xAxes: [{
              scaleLabel: {
                display: true,
                labelString: 'Epochs'
              }
            }]
          },
        datasets:[
          {
            label:"CNN",
            data:[0.9250906109809875,
                0.9663535952568054,
                0.9796844720840454,
                0.9870187640190125,
                0.9905816912651062,
                0.9925245046615601,
                0.9939228296279907,
                0.9944807887077332],
            backgroundColor:'transparent',
            borderColor:'yellow',
            tension:0.3,
            fill:true,
            pointStyle:'disc',
            pointBorderColor:'yellow',
            pointBackgroundColor:'#fff',
            showLine:true
          },
          {
            label:"SNN",
            data:[0.9262602925300598,
                0.9666090607643127,
                0.9763365983963013,
                0.9859364032745361,
                0.9917581677436829,
                0.9943665266036987,
                0.9956235885620117,
                0.9963765144348145],
            backgroundColor:'transparent',
            borderColor:'blue',
            tension:0.3,
            fill:true,
            pointStyle:'disc',
            pointBorderColor:'blue',
            pointBackgroundColor:'#fff',
            showLine:true
          },
          
          {
            label:"Bi-LSTM",
            data:[0.8667387962341309,
                0.959409236907959,
                0.9684644937515259,
                0.97312992811203,
                0.9774391055107117,
                0.9805852770805359,
                0.9824272394180298,
                0.9840608239173889],
            backgroundColor:'transparent',
            borderColor:'green',
            tension:0.3,
            fill:true,
            pointStyle:'disc',
            pointBorderColor:'green',
            pointBackgroundColor:'#fff',
            showLine:true
          }
        ]
      })
  return (
    <div className="linechart" style={{width:'600px', height:'215px'}}>
      <Line data={data}>Hello</Line>
    </div>
  )
}

export default Graph