import React from 'react'
import '../App.css'
import Imageslider from './Imageslider'


function Dashboard() {
  const slides = [
    { url: "http://localhost:3000/img1.png", title: "beach" },
    { url: "http://localhost:3000/img2.png", title: "boat" },
    { url: "http://localhost:3000/img3.png", title: "forest" },
    { url: "http://localhost:3000/img4.png", title: "city" },
    { url: "http://localhost:3000/img5.png", title: "italy" },
  ];
  return (
    <div className='dashboard'>
        <div className='about'></div>
        <div className='carousel'><Imageslider slides={slides}/></div>
    </div>
  )
}

export default Dashboard