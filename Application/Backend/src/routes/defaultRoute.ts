import { Express, Router, Request, Response } from "express";
import { configDotenv } from "dotenv";

configDotenv();

export const defaultRoute = Router();

defaultRoute.get('/', (req: Request, res: Response) => {
    res.send('<h1>Hello, Express.js Server</h1>');
});

