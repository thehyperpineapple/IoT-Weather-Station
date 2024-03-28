import { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { routes } from "./routes";


dotenv.config();

const express = require('express');
const app = express();
const port = Number(process.env.PORT) || 4000;

// app.get('/', (req: Request,res: Response) => {
//     res.send('<h1> Hello, Express.js Server </h1>');
// });

// routes
app.use('/', routes);

app.listen(port, () => {
    console.log(`[server]: Server is running on port ${port}`);
});
