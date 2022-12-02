<style>
  img {
    max-width: 50px;
    margin: 0 5px;
  }

  div {
    font-family: inherit;
    font-size: x-large;
    color: inherit;
    background: inherit;
    line-height: inherit;
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: center;
    margin: 0 0 1rem 0;
  }
</style>

<script>
  import { animals } from "./animals.js";

  let currentAnimal = Math.floor(Math.random() * animals.length);
  let animal = animals[currentAnimal];
</script>

<div>
  <img src="{animal.src}" alt="{animal.name}" />
  <span>OFRAK</span>
</div>
