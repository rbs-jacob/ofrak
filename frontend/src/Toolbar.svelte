<style>
  .hbox {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-start;
    align-items: stretch;
    width: 100%;
  }

  button {
    margin: 0.25em;
    padding: 0.75em 0.5em 0.5em 0.5em;
    background-color: var(--main-bg-color);
    color: var(--main-fg-color);
    border: 1px solid var(--main-fg-color);
    border-radius: 0;
    font-size: smaller;
    overflow: hidden;
    box-shadow: none;
  }

  button:hover,
  button:focus {
    outline: none;
    box-shadow: inset 1px 1px 0 var(--main-fg-color),
      inset -1px -1px 0 var(--main-fg-color);
  }

  button:active {
    box-shadow: inset 2px 2px 0 var(--main-fg-color),
      inset -2px -2px 0 var(--main-fg-color);
  }
</style>

<script>
  import Icon from "./Icon.svelte";

  export let toolbarButtons;
</script>

<div class="hbox">
  {#each toolbarButtons as button}
    <button
      on:click="{async (e) => {
        const oldIcon = button.iconUrl;
        const oldSymbol = button.symbol;
        button.iconUrl = '/icons/loading.svg';
        button.symbol = undefined;
        await button
          .onclick(e)
          .then((_) => {
            button.iconUrl = oldIcon;
            button.symbol = oldSymbol;
          })
          .catch((e) => {
            button.iconUrl = '/icons/error.svg';
            try {
              let errorObject = JSON.parse(e.message);
              alert(`${errorObject.type}: ${errorObject.message}`);
            } catch {
              alert(e);
            }
            console.error(e);
          });
      }}"
    >
      {#if button.symbol}
        {button.symbol}
      {:else if button.iconUrl}
        <Icon url="{button.iconUrl}" />
      {:else}
        {button.text}
      {/if}
    </button>
  {/each}
</div>
