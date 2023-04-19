import React from 'react'
import MovieIcon from '@mui/icons-material/Movie';
import TwitterIcon from '@mui/icons-material/Twitter';
import HomeIcon from '@mui/icons-material/Home';

export const Sidebardata = [
  {
    title: "Dashboard",
    icon: <HomeIcon/>,
    link: "/dashboard"
  },
  {
    title: "Twitter",
    icon: <TwitterIcon/>,
    link: "/twitter"
  },
  {
    title: "IMDB",
    icon: <MovieIcon/>,
    link: "/imdb"
  }
]

