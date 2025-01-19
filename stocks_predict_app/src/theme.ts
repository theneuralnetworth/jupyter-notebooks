import { createTheme } from "@mui/material/styles";

const lightTheme = createTheme({
  palette: {
    primary: {
      main: "#2951b5",
    },
    secondary: {
      main: "#f50057",
    }
  },
  typography: {
    fontFamily: "'Roboto', sans-serif",
  }
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: "#3f51b5",
    },
    secondary: {
      main: "#9c27b0",
    }
  },
  typography: {
    fontFamily: "'Roboto', sans-serif",
  }
});

export {lightTheme, darkTheme};
