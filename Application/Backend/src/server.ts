import { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { routes } from "./routes";
import cors from 'cors'; // Import cors as an ES module

dotenv.config();



const corsOptions: cors.CorsOptions = {
    origin: '*', 
    methods: ['GET', 'POST', 'PUT', 'DELETE'], 
    allowedHeaders: ['Content-Type', 'Authorization'], 
    credentials: true, 
    optionsSuccessStatus: 200 
  };
  

const express = require('express');
const app = express();
const port = Number(process.env.PORT) || 4000;



// routes
app.use('/', routes);
app.use(cors(corsOptions)); // Using the cors middleware with options
app.listen(port, () => {
    console.log(`[server]: Server is running on port ${port}`);
});
