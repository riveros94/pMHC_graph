---
title: "diagram"
format: html
---


```mermaid

flowchart TD
  APP@{ shape: circle, label: "app.py" }
  CG["create_graphs(manifest: Dict) -> List[Tuple]"]:::wide
  
  
  APP --> CG
  
  classDef wide padding:400px
```




