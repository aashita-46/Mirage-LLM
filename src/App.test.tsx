import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

test("launches the operational simulation console",()=>{render(<App/>);expect(screen.getByText(/See the conflict/i)).toBeInTheDocument();fireEvent.click(screen.getByRole("button",{name:/Launch Simulation/i}));expect(screen.getByText("SCENARIO")).toBeInTheDocument();expect(screen.getByRole("button",{name:/Simulate/i})).toBeInTheDocument();});
test("offers all five required scenarios",()=>{render(<App/>);fireEvent.click(screen.getByRole("button",{name:/Launch Simulation/i}));expect(screen.getAllByRole("option")).toHaveLength(5);});
test("starts and pauses deterministic simulation",()=>{render(<App/>);fireEvent.click(screen.getByRole("button",{name:/Launch Simulation/i}));fireEvent.click(screen.getByRole("button",{name:/Simulate/i}));expect(screen.getByText("RUNNING")).toBeInTheDocument();fireEvent.click(screen.getByRole("button",{name:/Pause/i}));expect(screen.getByText("PAUSED")).toBeInTheDocument();});
