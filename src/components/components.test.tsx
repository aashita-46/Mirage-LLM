import { render, screen } from "@testing-library/react";
import { ScoreGauge } from "./ScoreGauge";

test("renders the score gauge accessibly", () => {
  render(<ScoreGauge score={42.3}/>);
  expect(screen.getByLabelText(/42.3 out of 100/)).toBeInTheDocument();
});
