import { render, screen } from "@testing-library/react";
import { ScoreGauge } from "./ScoreGauge";
import { TokenHeatmap } from "./TokenHeatmap";

test("renders the score gauge accessibly", () => {
  render(<ScoreGauge score={42.3}/>);
  expect(screen.getByLabelText(/42.3 out of 100/)).toBeInTheDocument();
});
test("renders token risk", () => {
  render(<TokenHeatmap tokens={[{id:1,text:"Canberra",logprob:-.2,entropy:.4,normalisedUncertainty:.2,importanceWeight:1.35,weightedRisk:.27,category:"entity"}]}/>);
  expect(screen.getByText("Canberra")).toBeInTheDocument();
});
